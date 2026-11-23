"""Manual validation of the overlap flag: precision of flagged items, annotated blind.

export
  1. Population: overlap.csv rows with overlap > 0 only (strata partial and
     contaminated; clean items are not sampled). Stratified sample with equal
     counts where possible (leftover quota goes to the other stratum),
     shuffled, seeded.
  2. One snippet per item, by ONE rule for every item: the item's
     highest-count window (counts from the overlap cache, read-only; windows
     built exactly as overlap.py builds them; ties -> first window; a window
     that overlap.py replaced by a shorter fallback uses the fallback) is sent
     to infini-gram `search_docs` (maxnum 1). [[...]] marks the matched span.
  3. Retrieval: cached in its own file (--snippet-cache, resumable, atomic
     saves), per-call timeout, retries with exponential backoff, failures
     appended to --failed-log. An item whose snippet cannot be retrieved is
     DROPPED from the sample and counted in the summary; it is never shown
     with a blank snippet.
  4. Files: --out annotation sheet (rubric in its header lines; columns
     sample_id, corpus, benchmark, item_text, corpus_snippet,
     true_contamination, notes: no stratum, overlap score, window count or
     max_count); --key (sample_id -> stratum, overlap, counts; read only by
     score); <key>_summary.json (population, quota, kept and dropped counts).
score
  Joins the sheet with the key and reports per stratum and for all flagged
  items: precision of the overlap flag = yes / (yes + no), Wilson 95% CI,
  unsure count (excluded) and the number dropped before annotation.
  Scope: clean-item recall is not estimated (clean items are not sampled),
  and n-gram overlap cannot detect paraphrased contamination.

Usage:
  python validate_sample.py export --overlap overlap.csv --cache overlap_cache.json --n 200 --seed 0
  python validate_sample.py score --filled validation_sample.csv --key validation_key.csv
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

from analyze import assign_stratum
from atomicio import recover_tmp, write_json_atomic
from common import load_benchmark
from manifest import write_manifest
from overlap import API, INDEX, code_windows, windows

STRATA = ["partial", "contaminated"]
LABELS = {"yes", "no", "unsure"}
SHEET_COLS = ["sample_id", "corpus", "benchmark", "item_text", "corpus_snippet", "true_contamination", "notes"]
KEY_COLS = ["sample_id", "corpus", "benchmark", "id", "stratum", "overlap", "max_count", "n_words", "n_windows",
            "query_window", "query_window_count"]
RUBRIC = [
    "# RUBRIC for true_contamination: yes = the corpus snippet contains the question or answer near-verbatim;"
    " no = generic or boilerplate phrase only; unsure = excluded from precision and counted.",
    "# Every snippet was retrieved by the same rule: the item's highest-count window, sent to infini-gram"
    " search_docs; [[...]] marks the matched span.",
    "# Do not open the key file while annotating. Fill true_contamination with yes / no / unsure.",
]
SCOPE = ("Scope: this is the precision of the overlap flag on items with overlap > 0. Clean-item recall is "
         "out of scope (clean items are not sampled), and n-gram overlap misses paraphrased contamination.")


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def stratified_sample(df, n, seed, strata=STRATA):
    """Equal quota per stratum; quota a stratum cannot fill is reassigned."""
    groups = {s: df[df.stratum == s] for s in strata}
    quota = {s: 0 for s in strata}
    left = n
    while left > 0:
        open_ = [s for s in strata if quota[s] < len(groups[s])]
        if not open_:
            break
        share = max(1, left // len(open_))
        for s in open_:
            take = min(share, len(groups[s]) - quota[s], left)
            quota[s] += take
            left -= take
            if left == 0:
                break
    parts = [groups[s].sample(n=quota[s], random_state=seed) for s in strata if quota[s]]
    return pd.concat(parts).reset_index(drop=True), quota


def item_windows(text, benchmark, a):
    if benchmark in a.code_benchmarks:
        return code_windows(text, a.code_n, a.k, a.code_min_tokens)
    return windows(text, a.n_words, a.k, a.min_words)


def highest_count_window(ws, index, cache, fallback):
    """(window, count) with the highest cached count; ties -> first window. None if no count > 0."""
    best = None
    for w in ws:
        key = f"{index}|{w}"
        q = w
        if key not in cache and key in fallback:
            q = fallback[key]
        c = cache.get(f"{index}|{q}")
        if c is not None and c > 0 and (best is None or c > best[1]):
            best = (q, c)
    return best


class SnippetFetcher:
    """search_docs with a resumable cache, per-call timeout, retries with backoff and a failure log.
    Definitive answers (ok, no_documents, blocked_by_api, api_error) are cached; transient
    failures are logged and retried on the next run."""

    def __init__(self, api_url, cache_path, failed_log, max_disp_len, timeout, retries):
        self.api_url, self.cache_path, self.failed_log = api_url, cache_path, failed_log
        self.max_disp_len, self.timeout, self.retries = max_disp_len, timeout, retries
        self.cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                self.cache = json.load(f)
        self.cache.update(recover_tmp(cache_path) or {})  # newer data from a save whose replace was blocked
        self.calls = 0
        self.new = 0

    def save(self):
        write_json_atomic(self.cache_path, self.cache, ensure_ascii=False)  # retried on locks, else .tmp kept

    def fetch(self, index, query):
        key = f"{index}|{self.max_disp_len}|{query}"
        if key in self.cache:
            return self.cache[key]
        payload = {"index": index, "query_type": "search_docs", "query": query, "maxnum": 1,
                   "max_disp_len": self.max_disp_len}
        err = "no attempt made"
        for attempt in range(self.retries):
            self.calls += 1
            try:
                r = requests.post(self.api_url, json=payload, timeout=self.timeout)
                if r.status_code != 200:
                    err = f"HTTP {r.status_code}: {r.text[:200]}"
                else:
                    d = r.json()
                    if "error" in d:
                        res = {"status": "api_error", "snippet": "", "detail": str(d["error"])[:200]}
                    elif "documents" not in d:
                        err = f"unexpected response: {str(d)[:200]}"
                        res = None
                    elif not d["documents"]:
                        res = {"status": "no_documents", "snippet": ""}
                    elif d["documents"][0].get("blocked"):
                        res = {"status": "blocked_by_api", "snippet": ""}
                    else:
                        spans = d["documents"][0].get("spans") or []
                        text = "".join(f"[[{t}]]" if lab is not None else t for t, lab in spans)
                        res = {"status": "ok", "snippet": text} if text.strip() else {"status": "empty_snippet",
                                                                                     "snippet": ""}
                    if res is not None:
                        self.cache[key] = res
                        self.new += 1
                        if self.new % 10 == 0:
                            self.save()
                        return res
            except Exception as e:  # noqa: BLE001 - recorded in the failure log
                err = f"{type(e).__name__}: {e}"[:300]
            time.sleep(2 ** attempt)
        with open(self.failed_log, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{index}\t{err}\t{json.dumps(query, ensure_ascii=False)}\n")
        return {"status": "request_failed", "snippet": "", "detail": err}


def summary_path(key):
    return os.path.splitext(key)[0] + "_summary.json"


def write_sheet(path, sheet):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write("\n".join(RUBRIC) + "\n")
        sheet.to_csv(f, index=False)


def read_sheet(path):
    with open(path, encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    skip = 0
    while skip < len(lines) and lines[skip].lstrip('"').startswith("#"):
        skip += 1
    return pd.read_csv(path, skiprows=skip, keep_default_na=False, encoding="utf-8-sig")


def export(a):
    outs = [a.out, a.key, summary_path(a.key)]
    if len({os.path.abspath(p) for p in outs}) < 3:
        sys.exit("error: --out and --key must be different files")
    existing = [p for p in outs if os.path.exists(p)]
    if existing and not a.overwrite:
        sys.exit(f"error: {existing} already exist (annotations may be lost); pass --overwrite to replace them")
    ov = pd.read_csv(a.overlap).dropna(subset=["overlap"])
    if a.benchmarks:
        ov = ov[ov.benchmark.isin(a.benchmarks)]
    if a.corpora:
        ov = ov[ov.corpus.isin(a.corpora)]
    if "complete" in ov:
        ov = ov[ov.complete.astype(str).str.lower() != "false"]
    ov = ov[ov.overlap > 0].copy()
    if ov.empty:
        sys.exit("error: no rows with overlap > 0 left after filtering")
    ov["stratum"] = assign_stratum(ov.overlap, a.thr)
    sample, quota = stratified_sample(ov, a.n, a.seed)
    sample = sample.sample(frac=1, random_state=a.seed).reset_index(drop=True)
    print("population (overlap > 0):", ov.stratum.value_counts().to_dict(), " sample:", quota)

    with open(a.cache, encoding="utf-8") as f:  # read-only
        cache = json.load(f)
    fallback = {}
    if a.fallback_map and os.path.exists(a.fallback_map):
        with open(a.fallback_map, encoding="utf-8") as f:
            fallback = json.load(f)
    items = {bm: {it["id"]: it for it in load_benchmark(bm)} for bm in sample.benchmark.unique()}
    fetcher = SnippetFetcher(a.api_url, a.snippet_cache, a.failed_log, a.max_disp_len, a.timeout, a.retries)
    kept, dropped = [], []
    try:
        for _, r in sample.iterrows():
            it = items[r.benchmark][int(r.id)]
            index = INDEX[r.corpus]
            best = highest_count_window(item_windows(it["text"], r.benchmark, a), index, cache, fallback)
            if best is None:
                dropped.append({"stratum": r.stratum, "reason": "no_cached_window_with_count_above_0"})
                continue
            res = fetcher.fetch(index, best[0])
            if res["status"] != "ok":
                dropped.append({"stratum": r.stratum, "reason": res["status"]})
                continue
            kept.append({**r.to_dict(), "item_text": it["text"], "corpus_snippet": res["snippet"],
                         "query_window": best[0], "query_window_count": best[1]})
    finally:
        fetcher.save()
    if not kept:
        sys.exit(f"error: every sampled item was dropped: {pd.Series([d['reason'] for d in dropped]).value_counts().to_dict()}")
    k = pd.DataFrame(kept)
    k.insert(0, "sample_id", range(len(k)))
    sheet = k.assign(true_contamination="", notes="")[SHEET_COLS]
    write_sheet(a.out, sheet)
    k[[c for c in KEY_COLS if c in k.columns]].to_csv(a.key, index=False)
    dr = pd.DataFrame(dropped, columns=["stratum", "reason"])
    summ = {"population": {s: int((ov.stratum == s).sum()) for s in STRATA}, "quota": quota,
            "kept": {s: int((k.stratum == s).sum()) for s in STRATA},
            "dropped": {s: int((dr.stratum == s).sum()) for s in STRATA},
            "dropped_by_reason": dr.reason.value_counts().to_dict(),
            "http_calls": fetcher.calls, "snippets_cached": len(fetcher.cache)}
    with open(summary_path(a.key), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2)
    write_manifest("validate_sample.py export", a, inputs=[a.overlap, a.cache])
    print(f"wrote {len(sheet)} rows -> {a.out} (annotation sheet; rubric in its header lines)")
    print(f"wrote key -> {a.key} (do not open while annotating; read by the score mode)")
    print(f"dropped {len(dr)} sampled items whose snippet could not be retrieved: {summ['dropped_by_reason']} "
          f"-> {summary_path(a.key)}")


def score(a):
    df = read_sheet(a.filled)
    need = {"sample_id", "true_contamination"}
    if need - set(df.columns):
        sys.exit(f"error: {a.filled} lacks columns {sorted(need - set(df.columns))}")
    key = pd.read_csv(a.key)
    if {"sample_id", "stratum"} - set(key.columns):
        sys.exit(f"error: {a.key} lacks columns {sorted({'sample_id', 'stratum'} - set(key.columns))}")
    df["sample_id"] = pd.to_numeric(df.sample_id, errors="coerce")
    if set(df.sample_id) != set(key.sample_id):
        sys.exit(f"error: sample_ids differ between {a.filled} and {a.key} "
                 f"(only in sheet: {sorted(set(df.sample_id) - set(key.sample_id))[:10]}, "
                 f"only in key: {sorted(set(key.sample_id) - set(df.sample_id))[:10]})")
    df = df[["sample_id", "true_contamination"]].merge(key, on="sample_id", validate="one_to_one")
    lab = df.true_contamination.astype(str).str.strip().str.lower()
    bad = df[~lab.isin(LABELS | {""})]
    if len(bad):
        sys.exit(f"error: invalid true_contamination values {sorted(set(lab[bad.index]))} "
                 f"in sample_id(s) {bad.sample_id.tolist()[:20]}; use yes/no/unsure")
    df["lab"] = lab
    n_blank = int((lab == "").sum())
    if n_blank:
        print(f"warning: {n_blank} rows unlabelled; excluded")
    dropped = {}
    if os.path.exists(summary_path(a.key)):
        with open(summary_path(a.key), encoding="utf-8") as f:
            dropped = json.load(f).get("dropped", {})
    rows = []
    groups = [("all", df)] + ([(b, g) for b, g in df.groupby("benchmark")] if "benchmark" in df else [])
    for name, g in groups:
        for s, gs in [(s, g[g.stratum == s]) for s in STRATA] + [("all_flagged", g)]:
            yes, no, unsure = (int((gs.lab == v).sum()) for v in ("yes", "no", "unsure"))
            lo, hi = wilson(yes, yes + no)
            n_drop = (sum(dropped.values()) if s == "all_flagged" else dropped.get(s, 0)) if name == "all" else np.nan
            rows.append({"subset": name, "stratum": s, "judged": yes + no, "yes": yes, "no": no,
                         "unsure_excluded": unsure, "precision": yes / (yes + no) if yes + no else np.nan,
                         "wilson_lo": lo, "wilson_hi": hi, "dropped_before_annotation": n_drop})
    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))
    c = res[(res.subset == "all") & (res.stratum == "all_flagged")].iloc[0]
    print(f"\nPrecision of the overlap flag (overlap > 0): {c.precision:.3f} [{c.wilson_lo:.3f}, {c.wilson_hi:.3f}] "
          f"(judged {c.judged}, unsure excluded {c.unsure_excluded}, dropped before annotation "
          f"{int(c.dropped_before_annotation)})")
    print(SCOPE)
    res.to_csv(a.out, index=False)
    write_manifest("validate_sample.py score", a, inputs=[a.filled, a.key])
    print(f"saved -> {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    e = sub.add_parser("export")
    e.add_argument("--overlap", default="overlap.csv")
    e.add_argument("--cache", default="overlap_cache.json", help="overlap.py count cache (read only)")
    e.add_argument("--fallback-map", default="overlap_fallback.json", help="overlap.py fallback windows, if any")
    e.add_argument("--benchmarks", nargs="*", default=None, help="default: all in overlap.csv")
    e.add_argument("--corpora", nargs="*", default=None, help="default: all in overlap.csv")
    e.add_argument("--n", type=int, default=200)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--thr", type=float, default=0.5, help="must match analyze.py --thr")
    e.add_argument("--n-words", type=int, default=13, help="must match overlap.py --n")
    e.add_argument("--k", type=int, default=15, help="must match overlap.py --k")
    e.add_argument("--min-words", type=int, default=10)
    e.add_argument("--code-benchmarks", nargs="*", default=["humaneval"])
    e.add_argument("--code-n", type=int, default=32)
    e.add_argument("--code-min-tokens", type=int, default=16)
    e.add_argument("--max-disp-len", type=int, default=80, help="snippet length in tokens")
    e.add_argument("--api-url", default=API)
    e.add_argument("--timeout", type=float, default=60.0, help="seconds per search_docs call")
    e.add_argument("--retries", type=int, default=4)
    e.add_argument("--snippet-cache", default="validation_snippets.json")
    e.add_argument("--failed-log", default="validation_snippets_failed.log")
    e.add_argument("--out", default="validation_sample.csv", help="annotation sheet (blind)")
    e.add_argument("--key", default="validation_key.csv", help="sample_id -> stratum; read only by score")
    e.add_argument("--overwrite", action="store_true", help="replace existing sheet / key / summary files")
    s = sub.add_parser("score")
    s.add_argument("--filled", required=True)
    s.add_argument("--key", required=True, help="the key file written by export")
    s.add_argument("--out", default="validation_precision.csv")
    a = ap.parse_args()
    export(a) if a.mode == "export" else score(a)


if __name__ == "__main__":
    main()
