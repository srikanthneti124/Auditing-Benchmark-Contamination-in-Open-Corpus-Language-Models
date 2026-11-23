"""Step 1: n-gram overlap between benchmark items and pretraining corpora,
via the infini-gram API (no corpus download).

overlap = fraction of sampled n-word windows of the item that occur >=1 time
in the corpus. 0 = clean; 1 = every sampled window found.

Code benchmarks (--code-benchmarks, default: humaneval) use code-aware windows:
--code-n whitespace-separated tokens (default 32) whose query string is the
exact source slice, so newlines and indentation are kept. Text benchmarks
keep the original single-space-joined --n word windows (cache keys unchanged).
For code items the n_words column holds the whitespace-token count.

Queries run in parallel (--workers threads, one requests.Session each). Per
benchmark x corpus, the unique windows not yet in the cache are queried, then
rows are built from the cache. The cache is saved atomically every 200 new
entries and on exit (including Ctrl+C). HTTP 403/429/5xx and network errors
are retried with exponential backoff; queries that still fail are appended to
overlap_failed.log and are excluded from n_windows (never counted as zero).
If the optional `truststore` package is installed, HTTPS uses the operating
system certificate store.

File locks (OneDrive sync, antivirus): the rename of the .tmp cache file is
retried 10 times, 0.2 s apart (atomicio.py). If it still fails, the .tmp file
is kept, a warning goes to stderr and atomic_save_warnings.log, and the next
run merges the .tmp into the cache, so no counted window is lost.

Long-query rejections: HTTP 400/413/414/422 are not retried. For code
benchmarks the rejected window is replaced by its first half (in
whitespace tokens, source slice kept, but never fewer than --code-min-tokens
tokens), shortening again on a further rejection until it cannot get shorter. Each replacement is logged to
overlap_fallback.log and stored in overlap_fallback.json (original key ->
shorter window) so reruns and validate_sample.py reuse it; the fallback rate
per benchmark x corpus is written to overlap_fallback_summary.csv. Rejected
text-benchmark windows are logged to overlap_failed.log.

Completeness: expected unique windows per benchmark x corpus come from the
loaders (no API). After the queries, done / failed / remaining is printed per
pair. If any pair has remaining windows, overlap.csv is NOT written and the
exit code is 3 (run_overlap.bat then retries), unless --allow-partial is
passed: then overlap.csv gets an appended boolean column "complete" (per
benchmark x corpus) and overlap_partial_notice.txt lists what is missing.
--status prints the accounting from the cache and exits without any query.

Usage:
  python overlap.py --benchmarks gsm8k mmlu --corpora pile dolma --limit 1000 --workers 2
  python overlap.py --benchmarks humaneval --corpora pile dolma --code-n 32
  python overlap.py --benchmarks gsm8k mmlu humaneval --corpora pile dolma --status
"""
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import pandas as pd
import requests
from tqdm import tqdm

from atomicio import recover_tmp, replace_with_retry, write_json_atomic
from common import load_benchmark
from manifest import write_manifest

API = "https://api.infini-gram.io/"
# Verify current index names at https://infini-gram.io/api_doc
INDEX = {"pile": "v4_piletrain_llama", "dolma": "v4_dolma-v1_7_llama"}
CACHE_FILE = "overlap_cache.json"
FAILED_LOG = "overlap_failed.log"
FALLBACK_FILE = "overlap_fallback.json"
FALLBACK_LOG = "overlap_fallback.log"
FALLBACK_SUMMARY = "overlap_fallback_summary.csv"
NOTICE_FILE = "overlap_partial_notice.txt"
EXIT_INCOMPLETE = 3
RETRIES = 6  # attempts per window; backoff sleeps 1, 2, 4, ... seconds
RETRY_STATUS = {403, 429}  # plus every 5xx
REJECT_STATUS = {400, 413, 414, 422}  # query rejected (e.g. too long): not retried


def _subsample(allw, k):
    if len(allw) <= k:
        return allw
    if k <= 1:
        return allw[:k]
    step = (len(allw) - 1) / (k - 1)
    return [allw[round(i * step)] for i in range(k)]


def windows(text, n, k, min_words):
    w = text.split()
    if len(w) < n:
        return [" ".join(w)] if len(w) >= min_words else []
    allw = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    return _subsample(allw, k)


def code_windows(text, n, k, min_tokens):
    """Windows of n whitespace-separated tokens, returned as the original
    source slice (internal whitespace, newlines and indentation preserved)."""
    spans = [m.span() for m in re.finditer(r"\S+", text)]
    if len(spans) < n:
        return [text[spans[0][0]:spans[-1][1]]] if spans and len(spans) >= min_tokens else []
    allw = [text[spans[i][0]:spans[i + n - 1][1]] for i in range(len(spans) - n + 1)]
    return _subsample(allw, k)


class Counter:
    def __init__(self):
        self.cache = json.load(open(CACHE_FILE)) if os.path.exists(CACHE_FILE) else {}
        recovered = recover_tmp(CACHE_FILE)  # newer data from a save whose replace was blocked
        if recovered:
            self.cache.update(recovered)
        self.new = 0
        self.failed = 0
        self.rejected = set()  # cache-style keys rejected with REJECT_STATUS
        self.lock = threading.Lock()
        self.local = threading.local()

    def _session(self):
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        return self.local.session

    def count(self, index, q):
        key = f"{index}|{q}"
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        err = "no attempt made"
        for attempt in range(RETRIES):
            try:
                r = self._session().post(
                    API,
                    json={"index": index, "query_type": "count", "query": q},
                    timeout=30,
                )
                if r.status_code in REJECT_STATUS:
                    err = f"HTTP {r.status_code} (rejected, not retried): {r.text[:300]}"
                    with self.lock:
                        self.rejected.add(key)
                    break
                if r.status_code in RETRY_STATUS or r.status_code >= 500:
                    err = f"HTTP {r.status_code}: {r.text[:300]}"
                    time.sleep(2 ** attempt)
                    continue
                d = r.json()
                if "count" in d:
                    with self.lock:
                        self.cache[key] = d["count"]
                        self.new += 1
                        if self.new % 200 == 0:
                            self._save_locked()
                    time.sleep(0.05)
                    return d["count"]
                err = f"unexpected response (HTTP {r.status_code}): {str(d)[:300]}"
            except Exception as e:
                err = f"{type(e).__name__}: {e}"[:500]
            time.sleep(2 ** attempt)
        with self.lock:
            self.failed += 1
            with open(FAILED_LOG, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{index}\t{err}\t{json.dumps(q, ensure_ascii=False)}\n")
        return None

    def _save_locked(self):
        tmp = CACHE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.cache, f)
        replace_with_retry(tmp, CACHE_FILE)  # OneDrive / antivirus locks: retried, else .tmp kept + logged

    def save(self):
        with self.lock:
            self._save_locked()


def query_all(counter, index, todo, workers, desc):
    """Query windows in parallel. Waits in 1 s slices so Ctrl+C is delivered on Windows."""
    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        pending = {ex.submit(counter.count, index, w) for w in todo}
        with tqdm(total=len(pending), desc=desc) as bar:
            while pending:
                done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                for f in done:
                    f.result()
                bar.update(len(done))
    finally:
        ex.shutdown(wait=True, cancel_futures=True)


def load_json(path):
    data = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    return {**data, **(recover_tmp(path) or {})}


def save_json_atomic(path, obj):
    return write_json_atomic(path, obj, ensure_ascii=False)


def effective_count(cache, fallback, index, w):
    """Cached count of window w, or of its shorter fallback window; None if neither is known."""
    key = f"{index}|{w}"
    if key in cache:
        return cache[key]
    fb = fallback.get(key)
    return cache.get(f"{index}|{fb}") if fb is not None else None


def shorten(window, n_tokens):
    """First n_tokens whitespace tokens of window, as the original source slice."""
    spans = [m.span() for m in re.finditer(r"\S+", window)]
    return window[spans[0][0]:spans[min(n_tokens, len(spans)) - 1][1]]


def resolve_fallbacks(counter, index, ws, fallback, min_tokens, workers, desc):
    """Replace rejected windows by their first half (repeatedly) until accepted or < min_tokens."""
    pending = {w: len(re.findall(r"\S+", w)) for w in ws if f"{index}|{w}" in counter.rejected
               and f"{index}|{w}" not in fallback}
    while pending:
        target = {w: max(n // 2, min_tokens) for w, n in pending.items()}
        short = {w: shorten(w, target[w]) for w, n in pending.items() if target[w] < n}
        todo = sorted({s for s in short.values() if f"{index}|{s}" not in counter.cache})
        if todo:
            query_all(counter, index, todo, workers, f"{desc} fallback")
        nxt = {}
        for w, s in short.items():
            if f"{index}|{s}" in counter.cache:
                fallback[f"{index}|{w}"] = s
                with open(FALLBACK_LOG, "a", encoding="utf-8") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{index}\t{pending[w]} -> {target[w]} tokens"
                            f"\t{json.dumps(w, ensure_ascii=False)}\t{json.dumps(s, ensure_ascii=False)}\n")
            elif f"{index}|{s}" in counter.rejected:
                nxt[w] = target[w]
        pending = nxt


def failed_keys(path=FAILED_LOG):
    keys = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for ln in f:
                p = ln.rstrip("\n").split("\t")
                if len(p) >= 4:
                    try:
                        keys.add(f"{p[1]}|{json.loads(p[3])}")
                    except json.JSONDecodeError:
                        pass
    return keys


def accounting(plan, cache, fallback, failed):
    """done / failed / remaining unique windows per benchmark x corpus. plan: {(bm, corpus): windows}."""
    rows = []
    for (bm, corpus), ws in plan.items():
        index = INDEX[corpus]
        missing = [f"{index}|{w}" for w in ws if effective_count(cache, fallback, index, w) is None]
        n_fb = sum(f"{index}|{w}" in fallback for w in ws)
        rows.append({"benchmark": bm, "corpus": corpus, "expected": len(ws), "done": len(ws) - len(missing),
                     "failed": sum(k in failed for k in missing), "remaining": len(missing), "fallbacks": n_fb,
                     "fallback_rate": n_fb / len(ws) if ws else 0.0})
    return pd.DataFrame(rows, columns=["benchmark", "corpus", "expected", "done", "failed", "remaining",
                                       "fallbacks", "fallback_rate"])


def item_windows(items, bm, a):
    if bm in a.code_benchmarks:
        return [code_windows(it["text"], a.code_n, a.k, a.code_min_tokens) for it in items]
    return [windows(it["text"], a.n, a.k, a.min_words) for it in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=["gsm8k", "mmlu"])
    ap.add_argument("--corpora", nargs="+", default=["pile", "dolma"])
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=13, help="window length in words")
    ap.add_argument("--k", type=int, default=15, help="max windows per item")
    ap.add_argument("--min-words", type=int, default=10)
    ap.add_argument("--code-benchmarks", nargs="*", default=["humaneval"],
                    help="benchmarks that use code-aware windows")
    ap.add_argument("--code-n", type=int, default=32, help="code window length in whitespace tokens")
    ap.add_argument("--code-min-tokens", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2, help="parallel API queries")
    ap.add_argument("--out", default="overlap.csv")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write overlap.csv even if some benchmark x corpus is incomplete (adds column 'complete')")
    ap.add_argument("--status", action="store_true", help="print done / failed / remaining from the cache and exit")
    a = ap.parse_args()
    unknown = [c for c in a.corpora if c not in INDEX]
    if unknown:
        ap.error(f"unknown corpus {unknown}; choose from {sorted(INDEX)}")
    if a.k < 1 or a.n < 1 or a.code_n < 1:
        ap.error("--k, --n and --code-n must be >= 1")
    if a.workers < 1:
        ap.error("--workers must be >= 1")

    if a.status:
        plan = {}
        for bm in a.benchmarks:
            uniq = sorted({w for ws in item_windows(load_benchmark(bm, a.limit, a.seed), bm, a) for w in ws})
            plan.update({(bm, c): uniq for c in a.corpora})
        acct = accounting(plan, load_json(CACHE_FILE), load_json(FALLBACK_FILE), failed_keys())
        print(acct.to_string(index=False))
        print(f"total: {acct.expected.sum()} expected, {acct.done.sum()} done, {acct.failed.sum()} failed, "
              f"{acct.remaining.sum()} remaining")
        return

    write_manifest("overlap.py", a, inputs=[CACHE_FILE, FALLBACK_FILE])
    counter = Counter()
    fallback = load_json(FALLBACK_FILE)
    rows, plan = [], {}
    try:
        for bm in a.benchmarks:
            items = load_benchmark(bm, a.limit, a.seed)
            item_ws = item_windows(items, bm, a)
            uniq = sorted({w for ws in item_ws for w in ws})
            for corpus in a.corpora:
                index = INDEX[corpus]
                with counter.lock:
                    todo = [w for w in uniq if f"{index}|{w}" not in counter.cache and f"{index}|{w}" not in fallback]
                print(f"{bm}/{corpus}: {len(todo)} uncached windows")
                if todo:
                    query_all(counter, index, todo, a.workers, f"{bm}/{corpus}")
                if bm in a.code_benchmarks:
                    resolve_fallbacks(counter, index, uniq, fallback, a.code_min_tokens, a.workers, f"{bm}/{corpus}")
                plan[(bm, corpus)] = uniq
                for it, ws in zip(items, item_ws):
                    counts = [effective_count(counter.cache, fallback, index, w) for w in ws]
                    counts = [c for c in counts if c is not None]
                    rows.append(
                        {
                            "corpus": corpus,
                            "benchmark": bm,
                            "id": it["id"],
                            "n_words": len(it["text"].split()),
                            "n_windows": len(counts),
                            "overlap": (sum(c > 0 for c in counts) / len(counts)) if counts else float("nan"),
                            "max_count": max(counts) if counts else float("nan"),
                        }
                    )
    finally:
        counter.save()
        if fallback:
            save_json_atomic(FALLBACK_FILE, fallback)
        print(f"cache saved: {len(counter.cache)} entries ({counter.new} new, {counter.failed} failed"
              f"{' -> ' + FAILED_LOG if counter.failed else ''})")

    acct = accounting(plan, counter.cache, fallback, failed_keys())
    print("\n== completeness: unique windows per benchmark x corpus ==")
    print(acct.to_string(index=False))
    code = acct[acct.benchmark.isin(a.code_benchmarks)]
    if len(code):
        code.to_csv(FALLBACK_SUMMARY, index=False)
        print(f"fallback rate (code benchmarks) -> {FALLBACK_SUMMARY}: "
              + ", ".join(f"{r.benchmark}/{r.corpus} {r.fallbacks}/{r.expected} = {r.fallback_rate:.1%}"
                          for r in code.itertuples()))
    incomplete = acct[acct.remaining > 0]
    df = pd.DataFrame(rows)
    if len(incomplete) and not a.allow_partial:
        print(f"\nerror: {len(incomplete)} benchmark x corpus pair(s) incomplete ("
              + ", ".join(f"{r.benchmark}/{r.corpus}: {r.remaining} remaining, {r.failed} failed"
                          for r in incomplete.itertuples())
              + f"); {a.out} NOT written. Rerun to retry (failed windows are retried), "
                f"or pass --allow-partial.")
        sys.exit(EXIT_INCOMPLETE)
    if a.allow_partial:
        done = {(r.benchmark, r.corpus): r.remaining == 0 for r in acct.itertuples()}
        df["complete"] = [done[(b, c)] for b, c in zip(df.benchmark, df.corpus)]
        with open(NOTICE_FILE, "w", encoding="utf-8") as f:
            f.write(f"{a.out} written with --allow-partial at {time.strftime('%Y-%m-%d %H:%M:%S')}.\n"
                    f"Rows of incomplete benchmark x corpus pairs have complete=False; analyze.py excludes them "
                    f"by default.\n\n")
            f.write(incomplete.to_string(index=False) if len(incomplete) else "Nothing is missing: every pair is "
                    "complete.")
            f.write("\n")
        print(f"--allow-partial: {len(incomplete)} incomplete pair(s) listed in {NOTICE_FILE}")
    df.to_csv(a.out, index=False)
    print(f"saved {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
