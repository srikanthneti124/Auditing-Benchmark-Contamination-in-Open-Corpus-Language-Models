"""Manual verification sheets for the perturbations.

export-gsm8k      50 random perturbed GSM8K items (seeded) from perturbed_gsm8k.jsonl:
                  original / perturbed question, original / perturbed answer, number
                  mapping, and blank columns verdict (correct / wrong / unclear) and notes.
export-humaneval  30 random perturbed HumanEval items (seeded) from perturbed_humaneval.jsonl:
                  original / renamed prompt, rename map, blank verdict and notes. Also runs the
                  renamed canonical solution of each sampled item against the ORIGINAL tests
                  (execution.py subprocess with timeout; not a security sandbox) and reports
                  the pass rate (column canonical_passes_original_tests).
score             fraction judged correct = correct / (correct + wrong), Wilson 95% CI;
                  "unclear" is excluded and counted; lists the items judged wrong.

Usage:
  python check_perturbed_sample.py export-gsm8k --perturbed perturbed_gsm8k.jsonl --out perturb_check_gsm8k.csv
  python check_perturbed_sample.py export-humaneval --perturbed perturbed_humaneval.jsonl --out perturb_check_humaneval.csv
  python check_perturbed_sample.py score --filled perturb_check_gsm8k.csv
"""
import argparse
import json
import os
import random
import sys

import pandas as pd

from common import load_benchmark
from execution import humaneval_program, run_program
from manifest import write_manifest
from validate_sample import wilson

VERDICTS = {"correct", "wrong", "unclear"}
GSM8K_COLS = ["sample_id", "id", "original_question", "perturbed_question", "original_answer", "perturbed_answer",
              "number_mapping", "verdict", "notes"]
HE_COLS = ["sample_id", "id", "task_id", "entry_point", "rename_map", "original_prompt", "perturbed_prompt",
           "canonical_passes_original_tests", "test_status", "verdict", "notes"]


def read_jsonl(path):
    if not os.path.exists(path):
        sys.exit(f"error: {path} not found (run the perturbation script first)")
    with open(path, encoding="utf-8") as f:
        recs = [json.loads(ln) for ln in f if ln.strip()]
    if not recs:
        sys.exit(f"error: {path} is empty")
    return recs


def sample(recs, n, seed):
    return random.Random(seed).sample(recs, min(n, len(recs)))


def guard(path, overwrite):
    if os.path.exists(path) and not overwrite:
        sys.exit(f"error: {path} already exists (verdicts may be lost); pass --overwrite to replace it")


def export_gsm8k(a):
    guard(a.out, a.overwrite)
    recs = sample(read_jsonl(a.perturbed), a.n, a.seed)
    df = pd.DataFrame([{
        "sample_id": i, "id": r["id"], "original_question": r["original_question"],
        "perturbed_question": r["perturbed_question"], "original_answer": r["original_answer"],
        "perturbed_answer": r["perturbed_answer"],
        "number_mapping": json.dumps(r["number_mapping"], ensure_ascii=False), "verdict": "", "notes": "",
    } for i, r in enumerate(recs)])[GSM8K_COLS]
    df.to_csv(a.out, index=False, encoding="utf-8-sig")
    write_manifest("check_perturbed_sample.py export-gsm8k", a, inputs=[a.perturbed])
    print(f"wrote {len(df)} items -> {a.out}; fill verdict with correct / wrong / unclear "
          f"(correct = perturbed answer is right for the perturbed question)")


def export_humaneval(a):
    guard(a.out, a.overwrite)
    recs = sample(read_jsonl(a.perturbed), a.n, a.seed)
    tests = {it["id"]: it["test"] for it in load_benchmark("humaneval")}
    rows = []
    for i, r in enumerate(recs):
        res = run_program(humaneval_program(r["perturbed_prompt"], r["perturbed_canonical_solution"], tests[r["id"]],
                                            r["entry_point"]), a.timeout)
        rows.append({
            "sample_id": i, "id": r["id"], "task_id": r["task_id"], "entry_point": r["entry_point"],
            "rename_map": json.dumps(r["rename_map"], ensure_ascii=False), "original_prompt": r["original_prompt"],
            "perturbed_prompt": r["perturbed_prompt"], "canonical_passes_original_tests": res["status"] == "passed",
            "test_status": res["status"], "verdict": "", "notes": "",
        })
    df = pd.DataFrame(rows)[HE_COLS]
    df.to_csv(a.out, index=False, encoding="utf-8-sig")
    k, n = int(df.canonical_passes_original_tests.sum()), len(df)
    lo, hi = wilson(k, n)
    write_manifest("check_perturbed_sample.py export-humaneval", a, inputs=[a.perturbed])
    print(f"renamed canonical solution vs ORIGINAL tests: {k}/{n} pass ({k / n:.1%}, Wilson 95% CI "
          f"[{lo:.3f}, {hi:.3f}])")
    for r in df[~df.canonical_passes_original_tests].itertuples():
        print(f"  FAIL {r.task_id} (id {r.id}): {r.test_status}")
    print(f"wrote {n} items -> {a.out}; fill verdict with correct / wrong / unclear "
          f"(correct = renaming is consistent and meaning-preserving)")


def score(a):
    df = pd.read_csv(a.filled, keep_default_na=False, encoding="utf-8-sig")
    if {"id", "verdict"} - set(df.columns):
        sys.exit(f"error: {a.filled} lacks columns {sorted({'id', 'verdict'} - set(df.columns))}")
    v = df.verdict.astype(str).str.strip().str.lower()
    bad = df[~v.isin(VERDICTS | {""})]
    if len(bad):
        sys.exit(f"error: invalid verdicts {sorted(set(v[bad.index]))} for id(s) {bad.id.tolist()[:20]}; "
                 f"use correct / wrong / unclear")
    n_blank, n_unclear = int((v == "").sum()), int((v == "unclear").sum())
    k, w = int((v == "correct").sum()), int((v == "wrong").sum())
    lo, hi = wilson(k, k + w)
    frac = k / (k + w) if k + w else float("nan")
    if n_blank:
        print(f"warning: {n_blank} items without a verdict; excluded")
    print(f"judged correct: {k}/{k + w} = {frac:.3f} (Wilson 95% CI [{lo:.3f}, {hi:.3f}]); "
          f"unclear excluded: {n_unclear}")
    wrong = df[v == "wrong"]
    for r in wrong.itertuples():
        print(f"  WRONG id {r.id}: {getattr(r, 'notes', '')}")
    out = {"file": a.filled, "judged": k + w, "correct": k, "wrong": w, "unclear_excluded": n_unclear,
           "unlabelled": n_blank, "fraction_correct": frac, "wilson_lo": lo, "wilson_hi": hi,
           "wrong_ids": " ".join(str(i) for i in wrong.id)}
    if "canonical_passes_original_tests" in df:
        p = df.canonical_passes_original_tests.astype(str).str.lower() == "true"
        out["canonical_pass_rate"] = p.mean()
        print(f"renamed canonical solution passes original tests: {int(p.sum())}/{len(p)}")
    pd.DataFrame([out]).to_csv(a.out, index=False)
    write_manifest("check_perturbed_sample.py score", a, inputs=[a.filled])
    print(f"saved -> {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    g = sub.add_parser("export-gsm8k")
    g.add_argument("--perturbed", default="perturbed_gsm8k.jsonl")
    g.add_argument("--n", type=int, default=50)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default="perturb_check_gsm8k.csv")
    g.add_argument("--overwrite", action="store_true")
    h = sub.add_parser("export-humaneval")
    h.add_argument("--perturbed", default="perturbed_humaneval.jsonl")
    h.add_argument("--n", type=int, default=30)
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--timeout", type=float, default=10.0, help="seconds per test execution")
    h.add_argument("--out", default="perturb_check_humaneval.csv")
    h.add_argument("--overwrite", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--filled", required=True)
    s.add_argument("--out", default=None, help="default: <filled>_score.csv")
    a = ap.parse_args()
    if a.mode == "score" and a.out is None:
        a.out = os.path.splitext(a.filled)[0] + "_score.csv"
    {"export-gsm8k": export_gsm8k, "export-humaneval": export_humaneval, "score": score}[a.mode](a)


if __name__ == "__main__":
    main()
