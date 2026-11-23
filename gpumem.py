"""GPU memory safety shared by evaluate.py and itemloglik.py.

check_weights_fit      before a model loads: weight bytes for the chosen dtype
                       (parameter count from the model config, built on the
                       meta device, so nothing is downloaded but config.json)
                       versus free GPU memory; refuses above 90% of free.
generate_with_oom_retry  model.generate with one retry at half max_new_tokens
                       after a CUDA out-of-memory error; a second OOM raises
                       GenerationSkipped (the item is skipped, never scored).
track_peak             per benchmark: peak torch.cuda.max_memory_allocated,
                       seconds and items, appended to gpu_memory_log.csv.
"""
import csv
import os
import time
from contextlib import contextmanager

import torch

WEIGHTS_FRACTION = 0.9
GIB = 2 ** 30


class GenerationSkipped(Exception):
    pass


def append_csv(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def free_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def estimate_weight_bytes(name, revision=None, dtype=torch.float32, cache_dir=None):
    """(bytes, parameter count) of the model's weights in `dtype`; tied weights counted once."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(name, revision=revision, cache_dir=cache_dir)
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg)
    n = sum(p.numel() for p in m.parameters())
    return n * (torch.finfo(dtype).bits // 8), n


def check_weights_fit(name, revision, dtype, device, fraction=WEIGHTS_FRACTION, cache_dir=None):
    """Print the weights estimate; SystemExit if it exceeds `fraction` of free GPU memory."""
    if not str(device).startswith("cuda"):
        return None
    est, n = estimate_weight_bytes(name, revision, dtype, cache_dir=cache_dir)
    free, total = torch.cuda.mem_get_info(torch.device(device))
    print(f"weights estimate: {n / 1e9:.3f}B parameters x {torch.finfo(dtype).bits // 8} bytes ({dtype}) = "
          f"{est / GIB:.2f} GiB; free GPU memory {free / GIB:.2f} of {total / GIB:.2f} GiB")
    if est > fraction * free:
        raise SystemExit(f"error: weights alone ({est / GIB:.2f} GiB, {dtype}) exceed {fraction:.0%} of free GPU "
                         f"memory ({free / GIB:.2f} GiB) for {name}; free GPU memory, use a smaller model, or run it "
                         f"as a labelled float16 sensitivity model")
    return est


def generate_with_oom_retry(model, max_new_tokens, **kw):
    """(output, max_new_tokens used). CUDA OOM: free the cache, retry once with half the tokens;
    a second OOM raises GenerationSkipped."""
    try:
        with torch.no_grad():
            return model.generate(max_new_tokens=max_new_tokens, **kw), max_new_tokens
    except torch.cuda.OutOfMemoryError:
        pass
    free_cuda()
    small = max(1, max_new_tokens // 2)
    try:
        with torch.no_grad():
            return model.generate(max_new_tokens=small, **kw), small
    except torch.cuda.OutOfMemoryError:
        pass
    free_cuda()
    raise GenerationSkipped(f"CUDA out of memory at max_new_tokens={max_new_tokens} and again at {small}")


@contextmanager
def track_peak(path, script, model, benchmark, device):
    """Yields a dict; set rec["n_items"] (and optionally rec["n_skipped"]) before the block ends."""
    rec = {"n_items": 0, "n_skipped": 0}
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    t0 = time.time()
    yield rec
    sec = time.time() - t0
    peak = torch.cuda.max_memory_allocated(torch.device(device)) / 2 ** 20 if cuda else float("nan")
    reserved = torch.cuda.max_memory_reserved(torch.device(device)) / 2 ** 20 if cuda else float("nan")
    total = torch.cuda.get_device_properties(torch.device(device)).total_memory / 2 ** 20 if cuda else float("nan")
    n = rec["n_items"]
    append_csv(path, {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "script": script, "model": model, "benchmark": benchmark,
        "device": str(device), "n_items": n, "n_skipped": rec["n_skipped"], "seconds": round(sec, 2),
        "seconds_per_item": round(sec / n, 3) if n else "", "peak_allocated_mib": round(peak, 1),
        "peak_reserved_mib": round(reserved, 1), "device_total_mib": round(total, 1),
    })
    print(f"[{benchmark}] {n} items in {sec:.1f} s; peak GPU memory allocated {peak:.0f} MiB -> {path}")
