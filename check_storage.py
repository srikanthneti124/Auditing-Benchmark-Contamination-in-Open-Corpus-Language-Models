"""Disk check before any model download. Read-only (it downloads nothing).

Reports the Hugging Face cache path and size, free space on its drive, and for each model in
models.json (run order: reference last) the download still needed and PASS / FAIL: PASS when
that download fits in free space minus a safety margin (default 3 GB), counting the downloads
of the models before it that passed. A model whose size is unknown FAILS (never risk a
download that cannot finish).

Sizes: Hugging Face Hub metadata (weights files: *.safetensors, else *.bin) when reachable;
otherwise, or with --offline / HF_HUB_OFFLINE=1, "download_gb" from models.json. Bytes already
in the cache (snapshots/*/weights files) are subtracted.

--plan FILE writes "<model> <dtype> <max_new_tokens_gsm8k> <batch_size> <fits_8gb> <PASS|FAIL>"
lines for run_gpu_stages.bat. CHECK_STORAGE_FREE_GB overrides the measured free space (tests).

Usage:
  python check_storage.py
  python check_storage.py --cache-dir D:\\hf_cache --margin-gb 3
"""
import argparse
import glob
import os
import shutil
import sys

from modelcfg import load_models, load_run_settings

GB = 1e9
WEIGHT_EXT = (".safetensors", ".bin")


def hub_cache(cache_dir=None):
    if cache_dir:
        return cache_dir
    from huggingface_hub import constants
    return constants.HF_HUB_CACHE


def folder_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def free_bytes(path):
    p = os.path.abspath(path)
    while not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return shutil.disk_usage(p).free


def cached_weight_bytes(cache, repo):
    """{filename: bytes} of weights files in the newest-looking snapshot folders of repo."""
    d = os.path.join(cache, "models--" + repo.replace("/", "--"), "snapshots")
    out = {}
    for f in glob.glob(os.path.join(d, "*", "*")):
        if f.endswith(WEIGHT_EXT) and os.path.isfile(f):
            out[os.path.basename(f)] = max(out.get(os.path.basename(f), 0), os.path.getsize(f))
    return out


def hub_weight_bytes(repo, timeout=15):
    """{filename: bytes} from Hub metadata, or None if unreachable."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, files_metadata=True, timeout=timeout)
    except Exception as e:  # noqa: BLE001 - any failure means "use models.json"
        print(f"  (Hub metadata unavailable for {repo}: {type(e).__name__}; using models.json download_gb)")
        return None
    files = {s.rfilename: s.size or 0 for s in info.siblings}
    st = {f: b for f, b in files.items() if f.endswith(".safetensors")}
    return st or {f: b for f, b in files.items() if f.endswith(".bin")}


def plan(models_path="models.json", cache_dir=None, margin_gb=3.0, offline=False):
    import json
    with open(models_path, encoding="utf-8") as f:
        raw = json.load(f)
    cfg, run = load_models(models_path), load_run_settings(models_path)
    cache = hub_cache(cache_dir)
    free = float(os.environ["CHECK_STORAGE_FREE_GB"]) * GB if os.environ.get("CHECK_STORAGE_FREE_GB") else free_bytes(cache)
    offline = offline or os.environ.get("HF_HUB_OFFLINE") == "1"
    budget, reserved, rows = free - margin_gb * GB, 0.0, []
    for m, c in sorted(cfg.items(), key=lambda kv: kv[1]["role"] == "reference"):
        have = cached_weight_bytes(cache, m)
        hub = None if offline else hub_weight_bytes(m)
        if hub:
            total, source = sum(hub.values()), "hub"
            need = sum(max(0, b - have.get(f, 0)) for f, b in hub.items())
        else:
            gb = (raw.get(m) or {}).get("download_gb") if isinstance(raw.get(m), dict) else None
            total = gb * GB if isinstance(gb, (int, float)) else None
            source = "models.json" if total is not None else "unknown"
            need = None if total is None else max(0.0, total - sum(have.values()))
        if need is None:
            verdict, why = "FAIL", "download size unknown (add download_gb to models.json)"
        elif need <= budget - reserved:
            verdict, why = "PASS", ""
            reserved += need
        else:
            verdict, why = "FAIL", f"needs {need / GB:.2f} GB; {max(0, budget - reserved) / GB:.2f} GB left after margin"
        rows.append({"model": m, "dtype": c["dtype"], "total": total, "cached": sum(have.values()), "need": need,
                     "source": source, "verdict": verdict, "why": why, **run[m]})
    return {"cache": cache, "cache_size": folder_size(cache) if os.path.isdir(cache) else 0, "free": free,
            "margin": margin_gb * GB, "rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="models.json")
    ap.add_argument("--cache-dir", default=None, help="default: HF_HOME / ~/.cache/huggingface/hub")
    ap.add_argument("--margin-gb", type=float, default=3.0)
    ap.add_argument("--offline", action="store_true", help="do not query the Hub; use models.json download_gb")
    ap.add_argument("--plan", default=None, help="write the per-model plan for run_gpu_stages.bat")
    a = ap.parse_args(argv)
    p = plan(a.models, a.cache_dir, a.margin_gb, a.offline)
    print(f"Hugging Face cache: {p['cache']}  ({p['cache_size'] / GB:.2f} GB on disk)")
    print(f"free on its drive: {p['free'] / GB:.2f} GB; safety margin {p['margin'] / GB:.1f} GB")
    print(f"{'model':28s} {'dtype':8s} {'size GB':>8s} {'cached':>7s} {'needed':>7s}  source       verdict")
    for r in p["rows"]:
        f = lambda v: "?" if v is None else f"{v / GB:.2f}"  # noqa: E731
        print(f"{r['model']:28s} {r['dtype']:8s} {f(r['total']):>8s} {f(r['cached']):>7s} {f(r['need']):>7s}  "
              f"{r['source']:12s} {r['verdict']}{'  (' + r['why'] + ')' if r['why'] else ''}")
    if a.plan:
        with open(a.plan, "w", encoding="utf-8") as fh:
            for r in p["rows"]:
                fh.write(f"{r['model']} {r['dtype']} {r['max_new_tokens_gsm8k']} {r['batch_size']} "
                         f"{'yes' if r['fits_8gb'] else 'no'} {r['verdict']}\n")
    n_fail = sum(r["verdict"] == "FAIL" for r in p["rows"])
    print(f"{len(p['rows']) - n_fail} PASS, {n_fail} FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
