"""Shared benchmark loaders. Overlap detection and evaluation must use the
same (limit, seed) so item ids line up.

Item ids are dataset row indices (from sample_ids), so an item loaded with
limit=None has the same id as when it was loaded as part of a sample.
"""
import random
from datasets import load_dataset


def sample_ids(n, limit, seed):
    ids = list(range(n))
    if limit and limit < n:
        ids = sorted(random.Random(seed).sample(ids, limit))
    return ids


def load_gsm8k(limit=None, seed=0):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [
        {
            "id": i,
            "question": ds[i]["question"],
            "answer": ds[i]["answer"].split("####")[-1].strip().replace(",", ""),
            "solution": ds[i]["answer"],  # full reference solution with <<expr=result>> annotations
            "text": ds[i]["question"],
        }
        for i in sample_ids(len(ds), limit, seed)
    ]


def load_mmlu(limit=None, seed=0):
    ds = load_dataset("cais/mmlu", "all", split="test")
    return [
        {
            "id": i,
            "subject": ds[i]["subject"],
            "question": ds[i]["question"],
            "choices": ds[i]["choices"],
            "label": ds[i]["answer"],
            "text": ds[i]["question"],
        }
        for i in sample_ids(len(ds), limit, seed)
    ]


def load_humaneval(limit=None, seed=0):
    ds = load_dataset("openai/openai_humaneval", split="test")
    return [
        {
            "id": i,
            "task_id": ds[i]["task_id"],
            "prompt": ds[i]["prompt"],
            "canonical_solution": ds[i]["canonical_solution"],
            "test": ds[i]["test"],
            "entry_point": ds[i]["entry_point"],
            "text": ds[i]["prompt"],
        }
        for i in sample_ids(len(ds), limit, seed)
    ]


LOADERS = {"gsm8k": load_gsm8k, "mmlu": load_mmlu, "humaneval": load_humaneval}


def load_benchmark(name, limit=None, seed=0):
    if name not in LOADERS:
        raise ValueError(f"unknown benchmark {name!r}; choose from {sorted(LOADERS)}")
    return LOADERS[name](limit, seed)
