"""Model names in output files, and models.json.

Every output row names its model "<model>[@<revision>]@<dtype tag>", e.g.
EleutherAI/pythia-1.4b@fp32 or EleutherAI/pythia-160m@step1000@fp16, so rows
computed in different precisions are never mixed up.

models.json: {model: {"corpus": "pile" | "dolma" | null,
                      "role": "primary" | "reference" | "sensitivity",
                      "dtype": "float32" | "float16" | "bfloat16"}}
The older flat form {model: corpus, ..., "reference": model} is still read
(every model primary, float32).

Optional run settings per model (load_run_settings), set from measurements:
  "max_new_tokens_gsm8k" (default 256), "batch_size" (itemloglik, default 8),
  "fits_8gb" (default true; false makes run_gpu_stages.bat skip the model).

Usage (used by the .bat files):
  python modelcfg.py list        "<model> <dtype>" per line, reference last
  python modelcfg.py list-run    "<model> <dtype> <max_new_tokens_gsm8k> <batch_size> <fits_8gb yes|no>"
  python modelcfg.py reference   the reference model
"""
import json
import sys

DTYPE_TAGS = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}
TAG_DTYPES = {v: k for k, v in DTYPE_TAGS.items()}
ROLES = ("primary", "reference", "sensitivity")


def model_name(model, revision=None, dtype="float32"):
    if dtype not in DTYPE_TAGS:
        raise ValueError(f"unknown dtype {dtype!r}; choose from {sorted(DTYPE_TAGS)}")
    return model + (f"@{revision}" if revision else "") + "@" + DTYPE_TAGS[dtype]


def dtype_tag(name):
    """The dtype tag (fp32, fp16, bf16) of an output model name, or None."""
    head, _, tail = name.rpartition("@")
    return tail if head and tail in TAG_DTYPES else None


def strip_dtype(name):
    return name.rpartition("@")[0] if dtype_tag(name) else name


def base_model(name):
    return name.split("@", 1)[0]


def check_dtype(model, dtype):
    if dtype == "bfloat16" and "pythia" in model.lower():
        raise SystemExit("error: bfloat16 is not allowed for Pythia models (it shifted pythia-70m mean "
                         "log-likelihood by up to 2.6 nats); use float32, or float16 for a labelled sensitivity run")


def load_models(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise SystemExit(f"error: {path} must be a JSON object of model name -> settings")
    out = {}
    for k, v in raw.items():
        if isinstance(v, str):  # older flat form
            if k == "reference":
                out[v] = {"corpus": None, "role": "reference", "dtype": "float32"}
            else:
                out[k] = {"corpus": v, "role": "primary", "dtype": "float32"}
        elif (isinstance(v, dict) and v.get("role") in ROLES and v.get("dtype", "float32") in DTYPE_TAGS
              and (v.get("corpus") is None or isinstance(v.get("corpus"), str))):
            out[k] = {"corpus": v.get("corpus"), "role": v["role"], "dtype": v.get("dtype", "float32")}
        else:
            raise SystemExit(f"error: {path}: bad entry {k!r}: {v!r}; expected a corpus string or "
                             f"{{\"corpus\": ..., \"role\": one of {list(ROLES)}, \"dtype\": one of {sorted(DTYPE_TAGS)}}}")
    refs = [m for m, c in out.items() if c["role"] == "reference"]
    if len(refs) > 1:
        raise SystemExit(f"error: {path} names more than one reference model: {refs}")
    return out


RUN_DEFAULTS = {"max_new_tokens_gsm8k": 256, "batch_size": 8, "fits_8gb": True}


def load_run_settings(path):
    """{model: {max_new_tokens_gsm8k, batch_size, fits_8gb}} with defaults; flat-form files get defaults."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for m in load_models(path):
        v = raw.get(m) if isinstance(raw.get(m), dict) else {}
        s = {k: v.get(k, d) for k, d in RUN_DEFAULTS.items()}
        if not (isinstance(s["max_new_tokens_gsm8k"], int) and s["max_new_tokens_gsm8k"] > 0
                and isinstance(s["batch_size"], int) and s["batch_size"] > 0 and isinstance(s["fits_8gb"], bool)):
            raise SystemExit(f"error: {path}: bad run settings for {m!r}: {s}")
        out[m] = s
    return out


def lookup(name, cfg):
    """models.json settings for an output model name (with or without suffixes), or None."""
    for key in (name, strip_dtype(name), base_model(name)):
        if key in cfg:
            return cfg[key]
    return None


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "list"
    path = argv[2] if len(argv) > 2 else "models.json"
    cfg = load_models(path)
    if cmd == "list":
        for m, c in sorted(cfg.items(), key=lambda kv: kv[1]["role"] == "reference"):
            print(m, c["dtype"])
    elif cmd == "list-run":
        run = load_run_settings(path)
        for m, c in sorted(cfg.items(), key=lambda kv: kv[1]["role"] == "reference"):
            s = run[m]
            print(m, c["dtype"], s["max_new_tokens_gsm8k"], s["batch_size"], "yes" if s["fits_8gb"] else "no")
    elif cmd == "reference":
        refs = [m for m, c in cfg.items() if c["role"] == "reference"]
        if not refs:
            raise SystemExit("error: models.json has no model with role \"reference\"")
        print(refs[0])
    else:
        raise SystemExit(f"error: unknown command {cmd!r}; use list, list-run or reference")


if __name__ == "__main__":
    main(sys.argv)
