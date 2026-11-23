"""Per-item likelihood memorisation signals (work for near-chance models).

For each item, two texts are scored with the model:
  question           it["text"] (the same text overlap.py windows)
  question + answer  question + "\\n" + gold answer, where gold answer is
                     gsm8k     : the full reference solution as stored in the
                                 dataset (calculator annotations and "#### n")
                     mmlu      : the text of the correct choice
                     humaneval : the canonical solution (prompt + solution)
Scores per text (natural-log probabilities, averaged over tokens):
  ll_*    mean token log-likelihood
  mink_*  Min-K% Prob: mean of the lowest --mink fraction (default 0.2) of
          token log-probs (Shi et al., "Detecting Pretraining Data from Large
          Language Models")
A BOS token (EOS if the tokenizer has no BOS) is prepended when the tokenizer
does not add one, so every text token is scored. Texts are truncated to the
model's context length.

Forward passes are batched (--batch-size, right padding masked out; halved and
retried on CUDA out-of-memory) and run in float32 on every device unless
--dtype float16/bfloat16 is passed (bfloat16 is refused for Pythia);
log-softmax is always computed in fp32. --skip-existing resumes a run.
Before loading, the weights estimate is checked against free GPU memory
(gpumem.py); peak memory per benchmark is appended to --gpu-log.

Output (appended, like evaluate.py): model, benchmark, id, ll_question,
ll_question_answer, mink_question, mink_question_answer. The model name
carries the dtype, e.g. EleutherAI/pythia-1.4b@fp32 (modelcfg.model_name).

Usage:
  python itemloglik.py --model EleutherAI/pythia-1.4b --benchmarks gsm8k mmlu humaneval --limit 1000
  python itemloglik.py --model Qwen/Qwen2.5-1.5B --benchmarks gsm8k mmlu humaneval --limit 1000   # reference
"""
import argparse
import math
import os

import pandas as pd
import torch
from tqdm import tqdm

from common import load_benchmark
from evaluate import DTYPES, load_model, resolve_device
from gpumem import check_weights_fit, track_peak
from manifest import write_manifest
from modelcfg import check_dtype, model_name


def gold_text(benchmark, it):
    if benchmark == "gsm8k":
        return it["solution"]
    if benchmark == "mmlu":
        return it["choices"][it["label"]]
    if benchmark == "humaneval":
        return it["canonical_solution"]
    raise ValueError(f"no gold-answer format defined for benchmark {benchmark!r}")


def join_qa(benchmark, it):
    # HumanEval: the solution continues the prompt directly.
    sep = "" if benchmark == "humaneval" else "\n"
    return it["text"] + sep + gold_text(benchmark, it)


def encode(tok, text, max_len):
    ids = tok(text, add_special_tokens=True).input_ids
    start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    if start is not None and (not ids or ids[0] != start):
        ids = [start] + ids
    return ids[:max_len]


@torch.no_grad()
def token_logprobs(model, tok, text, max_len):
    ids = encode(tok, text, max_len)
    if len(ids) < 2:
        return []
    x = torch.tensor([ids], device=model.device)
    logits = model(x).logits[0, :-1].float()
    lp = torch.log_softmax(logits, -1).gather(1, x[0, 1:, None])[:, 0]
    return lp.tolist()


@torch.no_grad()
def batch_logprobs(model, tok, texts, max_len, batch_size):
    """Per-text token log-probs, computed in right-padded batches (padding is masked
    out and never scored). On CUDA out-of-memory the batch size is halved and the
    batch retried. Returns (list of lists in input order, final batch size)."""
    seqs = [encode(tok, t, max_len) for t in texts]
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]), reverse=True)
    out = [[] for _ in seqs]
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    pos = 0
    while pos < len(order):
        idx = [i for i in order[pos:pos + batch_size] if len(seqs[i]) >= 2]
        if not idx:
            pos += batch_size
            continue
        width = max(len(seqs[i]) for i in idx)
        x = torch.full((len(idx), width), pad, dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)
        for r, i in enumerate(idx):
            x[r, :len(seqs[i])] = torch.tensor(seqs[i])
            mask[r, :len(seqs[i])] = 1
        try:
            logits = model(input_ids=x.to(model.device), attention_mask=mask.to(model.device)).logits
            for r, i in enumerate(idx):
                n = len(seqs[i])
                lp = torch.log_softmax(logits[r, :n - 1].float(), -1)
                out[i] = lp.gather(1, x[r, 1:n, None].to(model.device))[:, 0].tolist()
        except torch.cuda.OutOfMemoryError:
            del x, mask
            if model.device.type == "cuda":
                torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            print(f"CUDA out of memory: retrying with --batch-size {batch_size}")
            continue
        pos += len(order[pos:pos + batch_size])
    return out, batch_size


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def min_k(v, k):
    if not v:
        return float("nan")
    n = max(1, math.ceil(k * len(v)))
    return mean(sorted(v)[:n])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--benchmarks", nargs="+", default=["gsm8k", "mmlu"])
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mink", type=float, default=0.2, help="fraction of lowest-probability tokens")
    ap.add_argument("--max-len", type=int, default=None, help="default: model context length (or 2048)")
    ap.add_argument("--out", default="itemloglik.csv")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip (model, benchmark, id) rows already present in --out")
    ap.add_argument("--device", default=None, help="cuda, cuda:0, cpu ... (default: cuda if available else cpu)")
    ap.add_argument("--batch-size", type=int, default=8, help="texts per forward pass (halved on CUDA OOM)")
    ap.add_argument("--dtype", default="float32", choices=list(DTYPES),
                    help="float32 on every device unless given; recorded in the model name (@fp32/@fp16/@bf16); "
                         "bfloat16 is refused for Pythia")
    ap.add_argument("--gpu-log", default="gpu_memory_log.csv", help="peak GPU memory per benchmark (appended)")
    ap.add_argument("--cache-dir", default=None,
                    help="Hugging Face model cache folder (default: HF_HOME or ~/.cache/huggingface)")
    a = ap.parse_args()
    if not 0 < a.mink <= 1:
        ap.error("--mink must be in (0, 1]")
    if a.batch_size < 1:
        ap.error("--batch-size must be >= 1")
    check_dtype(a.model, a.dtype)

    torch.manual_seed(a.seed)
    device = resolve_device(a.device)
    check_weights_fit(a.model, a.revision, DTYPES[a.dtype], device, cache_dir=a.cache_dir)
    tok, model = load_model(a.model, a.revision, device, dtype=DTYPES[a.dtype], cache_dir=a.cache_dir)
    print(f"model on {model.device}, dtype {model.dtype}")
    max_len = a.max_len or getattr(model.config, "max_position_embeddings", None) or 2048
    name = model_name(a.model, a.revision, a.dtype)
    write_manifest("itemloglik.py", a, model=a.model, dtype=a.dtype, revision=a.revision, device=device,
                   datasets=["openai/gsm8k", "cais/mmlu", "openai/openai_humaneval"])

    done = set()
    if a.skip_existing and os.path.exists(a.out):
        prev = pd.read_csv(a.out)
        prev = prev[prev.model == name]
        done = set(zip(prev.benchmark, prev.id))
        print(f"--skip-existing: {len(done)} rows for {name} already in {a.out}")

    rows, bs = [], a.batch_size
    for bm in a.benchmarks:
        items = [it for it in load_benchmark(bm, a.limit, a.seed) if (bm, it["id"]) not in done]
        with track_peak(a.gpu_log, "itemloglik.py", name, bm, device) as rec:
            for start in tqdm(range(0, len(items), 64), desc=bm):
                chunk = items[start:start + 64]
                q_lps, bs = batch_logprobs(model, tok, [it["text"] for it in chunk], max_len, bs)
                qa_lps, bs = batch_logprobs(model, tok, [join_qa(bm, it) for it in chunk], max_len, bs)
                for it, q, qa in zip(chunk, q_lps, qa_lps):
                    rows.append({
                        "model": name, "benchmark": bm, "id": it["id"],
                        "ll_question": mean(q), "ll_question_answer": mean(qa),
                        "mink_question": min_k(q, a.mink), "mink_question_answer": min_k(qa, a.mink),
                    })
            rec["n_items"] = len(items)
    if bs != a.batch_size:
        print(f"note: batch size reduced from {a.batch_size} to {bs} after CUDA out-of-memory")
    if not rows:
        print("nothing new to score")
        return
    df = pd.DataFrame(rows)
    df.to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
    print(df.groupby("benchmark")[["ll_question", "ll_question_answer", "mink_question"]].mean())


if __name__ == "__main__":
    main()
