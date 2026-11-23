"""Read-only health check for a running (or finished) overlap.py job.

Prints entries and nonzero counts per index, cache growth since the previous
call (state kept in --state), the failure count from overlap_failed.log, and
an ETA = (windows still uncached for the planned run) / (measured rate).
Never writes the cache. The cache may be mid-write by a non-atomic writer
(the old sequential overlap.py); a read that fails to parse is retried.

Usage:
  python check_overlap_health.py
  python check_overlap_health.py --cache overlap_cache.json --benchmarks gsm8k mmlu humaneval --limit 1000 --k 15
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

from overlap import FALLBACK_FILE, INDEX, code_windows, windows


def read_cache(path, tries=5):
    for i in range(tries):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            err = e
            time.sleep(2)
    sys.exit(f"error: {path} did not parse after {tries} tries ({err}); the writer may be mid-save, retry later")


def planned_windows(benchmarks, corpora, limit, seed, k, code_benchmarks):
    from common import load_benchmark

    keys = set()
    for bm in benchmarks:
        items = load_benchmark(bm, limit, seed)
        ws = {w for it in items for w in (code_windows(it["text"], 32, k, 16) if bm in code_benchmarks
                                          else windows(it["text"], 13, k, 10))}
        keys |= {f"{INDEX[c]}|{w}" for c in corpora for w in ws}
    return keys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="overlap_cache.json")
    ap.add_argument("--failed-log", default="overlap_failed.log")
    ap.add_argument("--state", default=".overlap_health_state.json", help="small file remembering the last check")
    ap.add_argument("--benchmarks", nargs="+", default=["gsm8k", "mmlu", "humaneval"])
    ap.add_argument("--corpora", nargs="+", default=["pile", "dolma"])
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--code-benchmarks", nargs="*", default=["humaneval"])
    ap.add_argument("--no-eta", action="store_true", help="skip loading benchmarks to compute remaining windows")
    a = ap.parse_args()
    if not os.path.exists(a.cache):
        sys.exit(f"error: {a.cache} not found")

    now = time.time()
    cache = read_cache(a.cache)
    mtime = os.path.getmtime(a.cache)
    per = Counter(k.split("|", 1)[0] for k in cache)
    nz = Counter(k.split("|", 1)[0] for k, v in cache.items() if v)
    print(f"cache {a.cache}: {len(cache)} entries, {os.path.getsize(a.cache) / 1e6:.1f} MB, "
          f"last saved {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))} ({(now - mtime) / 60:.1f} min ago)")
    for idx in sorted(per):
        print(f"  {idx:24s} entries {per[idx]:7d}   nonzero {nz[idx]:6d}")

    n_failed = 0
    if os.path.exists(a.failed_log):
        with open(a.failed_log, encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
        n_failed = len(lines)
        kinds = Counter(ln.split("\t")[2].split(":")[0] if ln.count("\t") >= 3 else "?" for ln in lines)
        print(f"failed windows logged: {n_failed}  {dict(kinds.most_common(5))}")
    else:
        print("failed windows logged: 0 (no overlap_failed.log)")

    rate = None
    prev = None
    if os.path.exists(a.state):
        with open(a.state, encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("cache") == os.path.abspath(a.cache) and mtime > prev["mtime"]:
            rate = (len(cache) - prev["entries"]) / ((mtime - prev["mtime"]) / 60)
            print(f"growth since last check: {len(cache) - prev['entries']:+d} entries over "
                  f"{(mtime - prev['mtime']) / 60:.1f} min of saves = {rate:.1f} entries/min")
        else:
            print("growth since last check: no new save since last check")
    else:
        print("growth since last check: first check (run again later to measure the rate)")
    with open(a.state, "w", encoding="utf-8") as f:
        json.dump({"cache": os.path.abspath(a.cache), "entries": len(cache), "mtime": mtime, "checked": now}, f)

    if a.no_eta:
        return
    keys = planned_windows(a.benchmarks, a.corpora, a.limit, a.seed, a.k, a.code_benchmarks)
    fallback = {}
    if os.path.exists(FALLBACK_FILE):  # windows overlap.py replaced by a shorter window count as done
        with open(FALLBACK_FILE, encoding="utf-8") as f:
            fallback = json.load(f)
    remaining = sum(k not in cache and not (k in fallback and f"{k.split('|', 1)[0]}|{fallback[k]}" in cache)
                    for k in keys)
    print(f"planned run ({' '.join(a.benchmarks)} x {' '.join(a.corpora)}, limit {a.limit}, k {a.k}): "
          f"{len(keys)} unique windows, {len(keys) - remaining} cached, {remaining} remaining")
    if rate and rate > 0:
        print(f"ETA at {rate:.1f}/min: {remaining / rate / 60:.1f} h")
    else:
        print("ETA: unknown until two checks see the cache grow")


if __name__ == "__main__":
    main()
