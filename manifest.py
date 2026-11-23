"""run_manifest.jsonl: one JSON line per data-producing script run.

Fields: timestamp, script, argv, args, seed, limit, versions (python, torch,
transformers, datasets, read from package metadata without importing them),
device name, dtype, model, revision, and the SHA-256 of each input file that
exists (missing inputs are recorded as null). Written to the current working
directory, like every other output, unless RUN_MANIFEST names another file.
"""
import hashlib
import json
import os
import platform
import sys
import time
from importlib import metadata

MANIFEST = "run_manifest.jsonl"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def versions():
    out = {"python": platform.python_version()}
    for pkg in ("torch", "transformers", "datasets", "pandas", "statsmodels"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def device_name(device=None):
    if device is None or not str(device).startswith("cuda"):
        return str(device or "cpu")
    torch = sys.modules.get("torch")
    try:
        return torch.cuda.get_device_name(torch.device(device)) if torch else str(device)
    except Exception:  # noqa: BLE001 - recording only
        return str(device)


def write_manifest(script, args=None, inputs=(), model=None, dtype=None, revision=None, device=None, **extra):
    a = vars(args) if args is not None and not isinstance(args, dict) else dict(args or {})
    rec = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "script": script,
        "argv": sys.argv[1:],
        "args": {k: v for k, v in a.items() if isinstance(v, (str, int, float, bool, list, type(None)))},
        "seed": a.get("seed"),
        "limit": a.get("limit"),
        "versions": versions(),
        "device": device_name(device),
        "dtype": dtype,
        "model": model,
        "revision": revision,
        "inputs": {p: (sha256(p) if p and os.path.isfile(p) else None) for p in inputs if p},
        **extra,
    }
    path = os.environ.get("RUN_MANIFEST", MANIFEST)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return rec
