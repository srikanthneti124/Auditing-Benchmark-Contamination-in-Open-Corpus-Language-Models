"""Preflight for the infini-gram API. Run it from YOUR OWN terminal before run_overlap.bat:

    python preflight_api.py

(a) applies truststore (operating-system certificates) if it is installed;
(b) positive control: a `count` query, on both indices, of a long sentence that is
    certain to occur in both corpora (the preamble of the US Constitution);
(c) one `search_docs` call with the same sentence;
(d) prints HTTP status, count / documents and latency for every call;
(e) exits with code 1 on any failure (HTTP error, 403, timeout, count 0, no documents).
Writes nothing.
"""
import argparse
import sys
import time

try:
    import truststore

    truststore.inject_into_ssl()
    TRUSTSTORE = "applied"
except ImportError:
    TRUSTSTORE = "not installed (Python's own certificate bundle is used)"

import requests

from overlap import API, INDEX

SENTENCE = ("We the People of the United States, in Order to form a more perfect Union, establish Justice, "
            "insure domestic Tranquility")


def call(api_url, payload, timeout):
    t = time.time()
    try:
        r = requests.post(api_url, json=payload, timeout=timeout)
        ms = (time.time() - t) * 1000
        try:
            body = r.json()
        except ValueError:
            body = {"_text": r.text[:200]}
        return r.status_code, body, ms, None
    except Exception as e:  # noqa: BLE001 - reported
        return None, {}, (time.time() - t) * 1000, f"{type(e).__name__}: {e}"[:300]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-url", default=API)
    ap.add_argument("--timeout", type=float, default=120.0)
    a = ap.parse_args(argv)
    print(f"truststore: {TRUSTSTORE}")
    print(f"API: {a.api_url}")
    failures = []
    for corpus, index in INDEX.items():
        status, body, ms, err = call(a.api_url, {"index": index, "query_type": "count", "query": SENTENCE}, a.timeout)
        count = body.get("count") if isinstance(body, dict) else None
        ok = status == 200 and isinstance(count, int) and count > 0
        print(f"[count] {corpus:5s} {index:22s} status={status} count={count} latency={ms:.0f} ms "
              f"{'OK' if ok else 'FAIL'}{' ' + err if err else ''}{'' if ok or err else ' ' + str(body)[:200]}")
        if not ok:
            failures.append(f"count on {index}")
    index = INDEX["pile"]
    status, body, ms, err = call(a.api_url, {"index": index, "query_type": "search_docs", "query": SENTENCE,
                                            "maxnum": 1, "max_disp_len": 40}, a.timeout)
    docs = body.get("documents") if isinstance(body, dict) else None
    ok = status == 200 and bool(docs)
    print(f"[search_docs] pile {index:22s} status={status} documents={len(docs) if docs else 0} latency={ms:.0f} ms "
          f"{'OK' if ok else 'FAIL'}{' ' + err if err else ''}{'' if ok or err else ' ' + str(body)[:200]}")
    if not ok:
        failures.append(f"search_docs on {index}")
    if failures:
        print(f"PREFLIGHT FAILED: {', '.join(failures)}. A 403 means the API is refusing this machine: "
              f"do not start run_overlap.bat.")
        return 1
    print("PREFLIGHT OK: both indices answer the positive control and search_docs returns a document.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
