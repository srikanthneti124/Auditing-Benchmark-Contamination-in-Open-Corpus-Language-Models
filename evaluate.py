"""Step 2: per-item correctness on raw and perturbed benchmarks.

MMLU     : k-shot, answer chosen by logits over A/B/C/D.
           Perturbation = answer-option permutation (items with "all/none of the
           above" style options are not permuted).
GSM8K    : 5-shot greedy generation, numeric match.
           Perturbation = numbers changed, gold answer recomputed from the
           reference solution's calculator annotations (perturb_gsm8k.py);
           only items that can be perturbed unambiguously get a perturbed row.
HumanEval: zero-shot greedy generation (pass@1), stop sequences
           ["\\ndef ", "\\nclass ", "\\nif __name__", "\\nprint("], tests run in
           a subprocess with a timeout (execution.py; NOT a security sandbox).
           Perturbation = alpha-renaming of parameters/locals
           (perturb_humaneval.py); only surviving items get a perturbed row.

--skip-existing skips (model, benchmark, id, variant) rows already in --out,
so an interrupted run can be resumed without duplicate rows.

--dtype is float32 on every device unless float16/bfloat16 is passed
explicitly; bfloat16 is refused for Pythia. The dtype is part of the model
name in every row (modelcfg.model_name), e.g. EleutherAI/pythia-1.4b@fp32.

GPU memory safety (gpumem.py): the weights estimate is printed before the
model loads and the run is refused if weights alone exceed 90% of free GPU
memory. A CUDA out-of-memory error during GSM8K/HumanEval generation is
retried once with half of max_new_tokens; a second OOM skips that item and
variant: it gets no row in --out (never counted as incorrect) and is logged
with a skipped_reason to --skip-log (retries are logged there too). Peak
memory per benchmark goes to --gpu-log.

Usage:
  python evaluate.py --model EleutherAI/pythia-1.4b --benchmarks gsm8k mmlu humaneval --limit 1000
  python evaluate.py --model allenai/OLMo-7B-hf --benchmarks gsm8k mmlu --limit 1000
  python evaluate.py --model Qwen/Qwen2.5-1.5B --benchmarks gsm8k mmlu --limit 1000   # reference model
"""
import argparse
import os
import random
import re
import time
from collections import defaultdict

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

import perturb_gsm8k
import perturb_humaneval
from common import load_benchmark
from execution import humaneval_program, run_program
from gpumem import GenerationSkipped, append_csv, check_weights_fit, generate_with_oom_retry, track_peak
from manifest import write_manifest
from modelcfg import check_dtype, model_name

LET = "ABCD"
NUM = re.compile(r"-?\d[\d,]*\.?\d*")
UNPERMUTABLE = re.compile(r"\b(above|both|neither|none of|all of)\b", re.I)
HUMANEVAL_STOP = ["\ndef ", "\nclass ", "\nif __name__", "\nprint("]


def resolve_device(device=None):
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"error: --device {device} requested but torch.cuda.is_available() is False "
                         f"(torch {torch.__version__})")
    return device


DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def load_model(name, revision=None, device=None, dtype=None, cache_dir=None):
    """cache_dir=None keeps the Hugging Face default (HF_HOME / ~/.cache/huggingface)."""
    dev = resolve_device(device)
    tok = AutoTokenizer.from_pretrained(name, revision=revision, cache_dir=cache_dir)
    if isinstance(dtype, str):
        dtype = DTYPES[dtype]
    if dtype is None:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype,
                                                 cache_dir=cache_dir).to(dev).eval()
    return tok, model


def fmt_q(q, choices):
    return q.strip() + "\n" + "".join(f"{l}. {c}\n" for l, c in zip(LET, choices)) + "Answer:"


def mmlu_prompt(shots, subject, q, choices):
    head = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
    body = "".join(fmt_q(s["question"], s["choices"]) + f" {LET[s['answer']]}\n\n" for s in shots)
    return head + body + fmt_q(q, choices)


def permute_choices(choices, label, seed, item_id):
    """Returns (permuted choices, new label), or None if the item is not permuted."""
    if any(UNPERMUTABLE.search(c) for c in choices):
        return None
    rng = random.Random(seed * 100003 + item_id)
    perm = list(range(4))
    rng.shuffle(perm)
    if perm == [0, 1, 2, 3]:
        perm = perm[1:] + perm[:1]
    return [choices[p] for p in perm], perm.index(label)


def run_mmlu(model, tok, items, k, seed, done=frozenset()):
    dev = defaultdict(list)
    for ex in load_dataset("cais/mmlu", "all", split="dev"):
        dev[ex["subject"]].append(ex)
    lid = [tok.encode(" " + l)[-1] for l in LET]

    @torch.no_grad()
    def predict(subject, q, choices):
        p = mmlu_prompt(dev[subject][:k], subject, q, choices)
        ids = tok(p, return_tensors="pt", truncation=True, max_length=2040).input_ids.to(model.device)
        return int(model(ids).logits[0, -1][lid].argmax())

    rows = []
    for it in tqdm(items, desc="mmlu"):
        if (it["id"], "raw") not in done:
            rows.append((it["id"], "raw", int(predict(it["subject"], it["question"], it["choices"]) == it["label"])))
        permuted = permute_choices(it["choices"], it["label"], seed, it["id"])
        if permuted is not None and (it["id"], "perturbed") not in done:
            ch, new_label = permuted
            rows.append((it["id"], "perturbed", int(predict(it["subject"], it["question"], ch) == new_label)))
    return rows


def extract_number(text):
    seg = text.split("####")[1] if "####" in text else text
    nums = NUM.findall(seg)
    if not nums:
        return None
    s = (nums[0] if "####" in text else nums[-1]).replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def guarded(events, item_id, variant, fn, *args):
    """fn(*args) -> (correct, max_new_tokens used, requested). Returns a row, or None if the item was
    skipped after a repeated CUDA OOM; retries and skips are appended to `events`."""
    try:
        correct, used, requested = fn(*args)
    except GenerationSkipped as e:
        events.append({"id": item_id, "variant": variant, "status": "skipped", "max_new_tokens_used": "",
                       "skipped_reason": str(e)})
        return None
    if used != requested:
        events.append({"id": item_id, "variant": variant, "status": "retried", "max_new_tokens_used": used,
                       "skipped_reason": ""})
    return (item_id, variant, correct)


def run_gsm8k(model, tok, items, shots=5, seed=0, done=frozenset(), max_new_tokens=256, events=None):
    events = [] if events is None else events
    train = load_dataset("openai/gsm8k", "main", split="train").select(range(shots))
    prefix = "".join(
        f"Question: {s['question']}\nAnswer: {re.sub(r'<<.*?>>', '', s['answer'])}\n\n" for s in train
    )

    def solve(question, gold):
        enc = tok(prefix + f"Question: {question}\nAnswer:", return_tensors="pt").to(model.device)
        out, used = generate_with_oom_retry(model, max_new_tokens, **enc, do_sample=False,
                                            pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).split("Question:")[0]
        pred = extract_number(text)
        return int(pred is not None and abs(pred - float(gold)) < 1e-4), used, max_new_tokens

    rows, n_pert = [], 0
    for it in tqdm(items, desc="gsm8k"):
        if (it["id"], "raw") not in done:
            rows.append(guarded(events, it["id"], "raw", solve, it["question"], it["answer"]))
        rec, _ = perturb_gsm8k.try_perturb(it, seed)
        if rec is not None:
            n_pert += 1
        if rec is not None and (it["id"], "perturbed") not in done:
            rows.append(guarded(events, it["id"], "perturbed", solve, rec["perturbed_question"],
                                rec["perturbed_answer"]))
    print(f"gsm8k: {n_pert}/{len(items)} items perturbed")
    print(f"gsm8k: {sum(r is None for r in rows)} item-variants skipped after CUDA out-of-memory")
    return [r for r in rows if r is not None]


class StopOnSubstrings(StoppingCriteria):
    """Stops (batch size 1) once the generated continuation contains any stop string."""

    def __init__(self, tok, prompt_len, stops):
        self.tok, self.prompt_len, self.stops = tok, prompt_len, stops

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tok.decode(input_ids[0, self.prompt_len:], skip_special_tokens=True)
        done = any(s in text for s in self.stops)
        return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)


def truncate_at_stop(text, stops):
    cut = min([text.find(s) for s in stops if s in text], default=len(text))
    return text[:cut]


def run_humaneval(model, tok, items, seed=0, max_new_tokens=512, timeout=10.0, reword_docstring=False,
                  done=frozenset(), events=None):
    events = [] if events is None else events
    max_ctx = getattr(model.config, "max_position_embeddings", 2048)

    def complete(prompt):
        enc = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=max(1, max_ctx - max_new_tokens)).to(model.device)
        n = enc.input_ids.shape[1]
        out, used = generate_with_oom_retry(
            model, max_new_tokens,
            **enc,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            stopping_criteria=StoppingCriteriaList([StopOnSubstrings(tok, n, HUMANEVAL_STOP)]),
        )
        return truncate_at_stop(tok.decode(out[0, n:], skip_special_tokens=True), HUMANEVAL_STOP), used

    def passes(prompt, test, entry):
        completion, used = complete(prompt)
        r = run_program(humaneval_program(prompt, completion, test, entry), timeout)
        return int(r["status"] == "passed"), used, max_new_tokens

    rows, n_pert = [], 0
    for it in tqdm(items, desc="humaneval"):
        if (it["id"], "raw") not in done:
            rows.append(guarded(events, it["id"], "raw", passes, it["prompt"], it["test"], it["entry_point"]))
        if (it["id"], "perturbed") in done:
            continue
        rec, _ = perturb_humaneval.try_perturb(it, seed, reword_docstring, timeout)
        if rec is not None:
            n_pert += 1
            rows.append(guarded(events, it["id"], "perturbed", passes, rec["perturbed_prompt"],
                                rec["perturbed_test"], it["entry_point"]))
    print(f"humaneval: {n_pert}/{len(items)} items perturbed")
    print(f"humaneval: {sum(r is None for r in rows)} item-variants skipped after CUDA out-of-memory")
    return [r for r in rows if r is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None, help="checkpoint, e.g. step100000 for Pythia")
    ap.add_argument("--benchmarks", nargs="+", default=["gsm8k", "mmlu"])
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--gsm8k-max-new-tokens", type=int, default=256)
    ap.add_argument("--humaneval-max-new-tokens", type=int, default=512)
    ap.add_argument("--humaneval-timeout", type=float, default=10.0, help="seconds per test execution")
    ap.add_argument("--humaneval-reword-docstring", action="store_true",
                    help="see perturb_humaneval.py --reword-docstring")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip (model, benchmark, id, variant) rows already present in --out")
    ap.add_argument("--device", default=None, help="cuda, cuda:0, cpu ... (default: cuda if available else cpu)")
    ap.add_argument("--dtype", default="float32", choices=list(DTYPES),
                    help="float32 on every device unless given; recorded in the model name (@fp32/@fp16/@bf16); "
                         "bfloat16 is refused for Pythia")
    ap.add_argument("--skip-log", default="generation_skips.csv",
                    help="items retried or skipped after CUDA out-of-memory (appended)")
    ap.add_argument("--gpu-log", default="gpu_memory_log.csv", help="peak GPU memory per benchmark (appended)")
    ap.add_argument("--cache-dir", default=None,
                    help="Hugging Face model cache folder (default: HF_HOME or ~/.cache/huggingface)")
    a = ap.parse_args()
    check_dtype(a.model, a.dtype)
    device = resolve_device(a.device)
    check_weights_fit(a.model, a.revision, DTYPES[a.dtype], device, cache_dir=a.cache_dir)

    tok, model = load_model(a.model, a.revision, device, a.dtype, cache_dir=a.cache_dir)
    tok.truncation_side = "left"
    print(f"model on {model.device}, dtype {model.dtype}")

    name = model_name(a.model, a.revision, a.dtype)
    write_manifest("evaluate.py", a, model=a.model, dtype=a.dtype, revision=a.revision, device=device,
                   datasets=["openai/gsm8k", "cais/mmlu", "openai/openai_humaneval"])
    existing = set()
    if a.skip_existing and os.path.exists(a.out):
        prev = pd.read_csv(a.out)
        prev = prev[prev.model == name]
        existing = set(zip(prev.benchmark, prev.id, prev.variant))
        print(f"--skip-existing: {len(existing)} rows for {name} already in {a.out}")
    all_rows = []
    for bm in a.benchmarks:
        items = load_benchmark(bm, a.limit, a.seed)
        done = frozenset((i, v) for b, i, v in existing if b == bm)
        events = []
        with track_peak(a.gpu_log, "evaluate.py", name, bm, device) as rec:
            if bm == "mmlu":
                rows = run_mmlu(model, tok, items, a.shots, a.seed, done)
            elif bm == "gsm8k":
                rows = run_gsm8k(model, tok, items, a.shots, a.seed, done, a.gsm8k_max_new_tokens, events)
            else:
                rows = run_humaneval(model, tok, items, a.seed, a.humaneval_max_new_tokens,
                                     a.humaneval_timeout, a.humaneval_reword_docstring, done, events)
            rec["n_items"] = len(rows)
            rec["n_skipped"] = sum(e["status"] == "skipped" for e in events)
        for e in events:
            append_csv(a.skip_log, {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "script": "evaluate.py",
                                    "model": name, "benchmark": bm, **e})
        all_rows += [{"model": name, "benchmark": bm, "id": i, "variant": v, "correct": c} for i, v, c in rows]

    if not all_rows:
        print("nothing new to evaluate")
        return
    df = pd.DataFrame(all_rows)
    df.to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
    print(df.groupby(["benchmark", "variant"]).correct.mean())


if __name__ == "__main__":
    main()
