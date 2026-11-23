"""GPU memory safety: simulated CUDA OOM during generation (retry, then skip; skipped items never
reach results.csv), weights-size refusal, peak-memory log, per-model run settings."""
import json
import os
import shutil
import subprocess
import sys

import pandas as pd
import pytest
import torch

import evaluate
import gpumem
import modelcfg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
GIB = 2 ** 30


class FakeModel:
    """generate() appends ' 18' to the prompt; raises CUDA OOM per a rule on (prompt text, max_new_tokens)."""

    def __init__(self, tok, oom_rule=lambda text, n: False):
        self.tok, self.oom_rule = tok, oom_rule
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.calls = []

        class Cfg:
            max_position_embeddings = 2048
        self.config = Cfg()

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=None, **kw):
        text = self.tok.decode(input_ids[0])
        self.calls.append(max_new_tokens)
        if self.oom_rule(text, max_new_tokens):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        add = torch.tensor([self.tok.encode(" 18\n")])
        return torch.cat([input_ids, add], dim=1)


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")


def test_generate_retries_once_with_half_tokens(tok):
    m = FakeModel(tok, oom_rule=lambda text, n: n > 200)
    enc = tok("Question: x", return_tensors="pt")
    out, used = gpumem.generate_with_oom_retry(m, 256, **enc)
    assert used == 128 and m.calls == [256, 128]
    m2 = FakeModel(tok, oom_rule=lambda text, n: True)
    with pytest.raises(gpumem.GenerationSkipped, match="max_new_tokens=256 and again at 128"):
        gpumem.generate_with_oom_retry(m2, 256, **enc)
    assert m2.calls == [256, 128]  # exactly one retry


def gsm8k_items(n=6):
    from common import load_benchmark
    return load_benchmark("gsm8k", n, 0)


def test_run_gsm8k_skips_oom_items_and_logs_retries(tok):
    items = gsm8k_items()
    skip_q, retry_q = items[0]["question"][:40], items[1]["question"][:40]
    rule = lambda text, n: (skip_q in text) or (retry_q in text and n == 256)  # noqa: E731
    events = []
    rows = evaluate.run_gsm8k(FakeModel(tok, rule), tok, items, shots=2, seed=0, events=events)
    ids = {(i, v) for i, v, _ in rows}
    assert (items[0]["id"], "raw") not in ids  # skipped: no row at all, not an incorrect row
    assert (items[1]["id"], "raw") in ids  # retried with half the tokens: scored normally
    skipped = [e for e in events if e["status"] == "skipped"]
    assert {(e["id"], e["variant"]) for e in skipped} >= {(items[0]["id"], "raw")}
    assert all("CUDA out of memory" in e["skipped_reason"] for e in skipped)
    assert {(e["id"], e["variant"], e["max_new_tokens_used"]) for e in events if e["status"] == "retried"} >= {
        (items[1]["id"], "raw", 128)}
    assert len(rows) + len(skipped) == sum(1 for _ in rows) + len(skipped)


def test_evaluate_main_excludes_skipped_items_from_results(tmp_path, monkeypatch, tok, capsys):
    items = gsm8k_items(3)  # the same seeded sample evaluate.py --limit 3 loads
    skip_q = items[1]["question"][:40]
    fake = FakeModel(tok, lambda text, n: skip_q in text)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(evaluate, "load_model", lambda *a, **k: (tok, fake))
    monkeypatch.setattr(evaluate, "check_weights_fit", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--model", "EleutherAI/pythia-70m", "--benchmarks", "gsm8k",
                                      "--limit", "3", "--shots", "2", "--device", "cpu", "--out", "r.csv"])
    evaluate.main()
    out = capsys.readouterr().out
    res = pd.read_csv("r.csv")
    assert items[1]["id"] not in set(res[res.variant == "raw"].id)
    assert set(res[res.variant == "raw"].id) == {items[0]["id"], items[2]["id"]}
    skips = pd.read_csv("generation_skips.csv")
    assert set(skips.columns) >= {"model", "benchmark", "id", "variant", "status", "skipped_reason"}
    s = skips[skips.status == "skipped"]
    assert (items[1]["id"], "raw") in set(zip(s.id, s.variant))
    assert (s.model == "EleutherAI/pythia-70m@fp32").all()
    assert "item-variants skipped after CUDA out-of-memory" in out
    log = pd.read_csv("gpu_memory_log.csv")
    assert list(log.benchmark) == ["gsm8k"] and log.n_skipped.iloc[0] == len(s) and log.n_items.iloc[0] == len(res)
    # --skip-existing retries the skipped item later (it has no row)
    monkeypatch.setattr(evaluate, "load_model", lambda *a, **k: (tok, FakeModel(tok)))
    monkeypatch.setattr(sys, "argv", sys.argv + ["--skip-existing"])
    evaluate.main()
    res2 = pd.read_csv("r.csv")
    assert set(res2[res2.variant == "raw"].id) == {it["id"] for it in items}
    assert not res2.duplicated(["model", "benchmark", "id", "variant"]).any()


def test_weights_size_refusal(monkeypatch, capsys):
    monkeypatch.setattr(gpumem, "estimate_weight_bytes", lambda *a, **k: (int(5.8 * GIB), 1_450_000_000))
    monkeypatch.setattr(gpumem.torch.cuda, "mem_get_info", lambda *a, **k: (int(6.0 * GIB), int(8.0 * GIB)))
    with pytest.raises(SystemExit, match=r"weights alone \(5.80 GiB.*exceed 90% of free GPU memory \(6.00 GiB\)"):
        gpumem.check_weights_fit("big/model", None, torch.float32, "cuda")
    assert "weights estimate: 1.450B parameters x 4 bytes" in capsys.readouterr().out
    monkeypatch.setattr(gpumem.torch.cuda, "mem_get_info", lambda *a, **k: (int(7.0 * GIB), int(8.0 * GIB)))
    assert gpumem.check_weights_fit("big/model", None, torch.float32, "cuda") == int(5.8 * GIB)


def test_weights_check_skipped_on_cpu(monkeypatch):
    monkeypatch.setattr(gpumem, "estimate_weight_bytes", lambda *a, **k: pytest.fail("must not estimate on CPU"))
    assert gpumem.check_weights_fit("m", None, torch.float32, "cpu") is None


def test_evaluate_refuses_before_loading(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gpumem, "estimate_weight_bytes", lambda *a, **k: (int(9 * GIB), 2_400_000_000))
    monkeypatch.setattr(gpumem.torch.cuda, "mem_get_info", lambda *a, **k: (int(6 * GIB), int(8 * GIB)))
    monkeypatch.setattr(evaluate, "resolve_device", lambda d=None: "cuda")
    monkeypatch.setattr(evaluate, "load_model", lambda *a, **k: pytest.fail("model must not be loaded"))
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--model", "EleutherAI/pythia-70m", "--benchmarks", "mmlu",
                                      "--out", "r.csv"])
    with pytest.raises(SystemExit, match="exceed 90% of free GPU memory"):
        evaluate.main()
    assert not (tmp_path / "r.csv").exists()


def test_weight_estimate_matches_loaded_model():
    from transformers import AutoModelForCausalLM
    est, n = gpumem.estimate_weight_bytes("EleutherAI/pythia-70m", None, torch.float32)
    m = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m", dtype=torch.float32)
    assert n == sum(p.numel() for p in m.parameters()) and est == 4 * n
    assert gpumem.estimate_weight_bytes("EleutherAI/pythia-70m", None, torch.float16)[0] == 2 * n


def test_track_peak_appends_row(tmp_path):
    path = tmp_path / "g.csv"
    with gpumem.track_peak(str(path), "s.py", "m@fp32", "gsm8k", "cpu") as rec:
        rec["n_items"] = 4
    with gpumem.track_peak(str(path), "s.py", "m@fp32", "mmlu", "cpu") as rec:
        rec["n_items"] = 0
    log = pd.read_csv(path)
    assert list(log.benchmark) == ["gsm8k", "mmlu"] and list(log.n_items) == [4, 0]
    assert {"peak_allocated_mib", "seconds_per_item", "model", "script"} <= set(log.columns)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_track_peak_measures_cuda_allocations(tmp_path):
    path = tmp_path / "g.csv"
    with gpumem.track_peak(str(path), "s.py", "m", "b", "cuda") as rec:
        x = torch.empty(64 * 2 ** 20, dtype=torch.uint8, device="cuda")  # 64 MiB
        rec["n_items"] = 1
        del x
    assert pd.read_csv(path).peak_allocated_mib.iloc[0] >= 64


# ---------------------------------------------------------------- run settings in models.json
def test_run_settings_defaults_and_values(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({
        "a/x": {"corpus": "pile", "role": "primary", "dtype": "float32", "max_new_tokens_gsm8k": 128, "batch_size": 2,
                "fits_8gb": False},
        "b/y": {"corpus": None, "role": "reference", "dtype": "float32"}}))
    s = modelcfg.load_run_settings(p)
    assert s["a/x"] == {"max_new_tokens_gsm8k": 128, "batch_size": 2, "fits_8gb": False}
    assert s["b/y"] == modelcfg.RUN_DEFAULTS
    assert modelcfg.load_models(p)["a/x"] == {"corpus": "pile", "role": "primary", "dtype": "float32"}
    p.write_text(json.dumps({"a/x": {"corpus": "pile", "role": "primary", "batch_size": 0}}))
    with pytest.raises(SystemExit, match="bad run settings"):
        modelcfg.load_run_settings(p)


def test_project_models_json_records_measurement():
    s = modelcfg.load_run_settings(os.path.join(ROOT, "models.json"))
    assert s["EleutherAI/pythia-410m"] == {"max_new_tokens_gsm8k": 256, "batch_size": 8, "fits_8gb": True}


@pytest.mark.skipif(sys.platform != "win32", reason="cmd.exe")
def test_gpu_stages_skip_model_that_does_not_fit(tmp_path):
    for f in ("run_gpu_stages.bat", "modelcfg.py", "check_storage.py"):
        shutil.copy(os.path.join(ROOT, f), tmp_path / f)
    (tmp_path / "models.json").write_text(json.dumps({  # download_gb: lets the storage check size models offline
        "EleutherAI/pythia-1.4b": {"corpus": "pile", "role": "primary", "dtype": "float32", "fits_8gb": False,
                                   "download_gb": 2.93},
        "EleutherAI/pythia-410m": {"corpus": "pile", "role": "primary", "dtype": "float32",
                                   "max_new_tokens_gsm8k": 192, "batch_size": 4, "download_gb": 0.91},
        "Qwen/Qwen2.5-1.5B": {"corpus": None, "role": "reference", "dtype": "float32", "download_gb": 3.09}}))
    env = dict(os.environ, HF_HUB_OFFLINE="1", CHECK_STORAGE_FREE_GB="1000")  # storage check: fixed, offline
    p = subprocess.run(["cmd", "/c", str(tmp_path / "run_gpu_stages.bat"), "--dry-run"], cwd=tmp_path,
                       capture_output=True, text=True, timeout=120, env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    assert 'SKIPPED EleutherAI/pythia-1.4b: models.json says "fits_8gb": false' in p.stdout
    cmds = [ln for ln in p.stdout.splitlines() if ln.startswith(">")]
    assert not [c for c in cmds if "pythia-1.4b" in c]
    assert any("pythia-410m" in c and "--gsm8k-max-new-tokens 192" in c for c in cmds)
    assert any("pythia-410m" in c and "itemloglik.py" in c and "--batch-size 4" in c for c in cmds)
    assert any("Qwen" in c and "--gsm8k-max-new-tokens 256" in c for c in cmds)
