"""dtype-suffixed model names, models.json roles, sensitivity separation,
regression skip reporting, blinded validation export, and the GPU/analysis
.bat split. Analysis tests run in tmp_path on synthetic data."""
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

import analyze
import modelcfg
import report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY, OLMO, P28, QWEN = "EleutherAI/pythia-410m", "allenai/OLMo-1B-0724-hf", "EleutherAI/pythia-2.8b", "Qwen/Qwen2.5-1.5B"
CFG = {PY: {"corpus": "pile", "role": "primary", "dtype": "float32"},
       OLMO: {"corpus": "dolma", "role": "primary", "dtype": "float32"},
       P28: {"corpus": "pile", "role": "sensitivity", "dtype": "float16"},
       QWEN: {"corpus": None, "role": "reference", "dtype": "float32"}}
ALL = [f"{PY}@fp32", f"{OLMO}@fp32", f"{P28}@fp16", f"{QWEN}@fp32"]
N = 90
SCORES = ["ll_question", "ll_question_answer", "mink_question", "mink_question_answer"]


@pytest.fixture(autouse=True)
def clean_globals():
    analyze.MODEL_CORPUS.clear()
    analyze.MODEL_CFG.clear()
    yield
    analyze.MODEL_CORPUS.clear()
    analyze.MODEL_CFG.clear()


def overlap_df():
    rows = []
    for i in range(N):
        for c in ("pile", "dolma"):
            o = [0.0, 0.3, 0.8][i % 3]
            rows.append({"corpus": c, "benchmark": "mmlu", "id": i, "n_words": 10 + i % 7, "n_windows": 10,
                         "overlap": o, "max_count": int(o * 10)})
        # gsm8k x pile is clean-only (one partial item, nothing contaminated); dolma is mixed
        rows.append({"corpus": "pile", "benchmark": "gsm8k", "id": i, "n_words": 20 + i % 5, "n_windows": 15,
                     "overlap": 0.1 if i == 0 else 0.0, "max_count": int(i == 0)})
        rows.append({"corpus": "dolma", "benchmark": "gsm8k", "id": i, "n_words": 20 + i % 5, "n_windows": 15,
                     "overlap": [0.0, 0.6][i % 2], "max_count": 3 * (i % 2)})
    return pd.DataFrame(rows)


def results_df(names, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame([{"model": m, "benchmark": b, "id": i, "variant": v, "correct": int(rng.random() < 0.5)}
                         for m in names for b in ("mmlu", "gsm8k") for i in range(N) for v in ("raw", "perturbed")])


def loglik_df(names, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame([{"model": m, "benchmark": b, "id": i, **{s: -rng.random() * 5 for s in SCORES}}
                         for m in names for b in ("mmlu", "gsm8k") for i in range(N)])


def run_analyze(d, monkeypatch, names, *extra, cfg=CFG, ref=QWEN):
    d.mkdir(exist_ok=True)
    monkeypatch.chdir(d)
    results_df(names).to_csv("results.csv", index=False)
    loglik_df(names).to_csv("itemloglik.csv", index=False)
    overlap_df().to_csv("overlap.csv", index=False)
    argv = ["analyze.py", "--results", "results.csv", "--overlap", "overlap.csv", "--loglik", "itemloglik.csv"]
    if ref:
        argv += ["--ref-model", ref]
    if cfg is not None:
        (d / "models.json").write_text(json.dumps(cfg))
        argv += ["--model-corpus-map", "models.json"]
    monkeypatch.setattr(sys, "argv", argv + list(extra))
    analyze.MODEL_CORPUS.clear()
    analyze.MODEL_CFG.clear()
    analyze.main()


# ------------------------------------------------------------------ modelcfg
def test_model_name_carries_dtype():
    assert modelcfg.model_name(PY, None, "float32") == f"{PY}@fp32"
    assert modelcfg.model_name(PY, "step1000", "float16") == f"{PY}@step1000@fp16"
    assert modelcfg.dtype_tag(f"{PY}@step1000@fp16") == "fp16"
    assert modelcfg.dtype_tag(f"{PY}@step1000") is None
    assert modelcfg.strip_dtype(f"{PY}@step1000@fp16") == f"{PY}@step1000"
    assert modelcfg.base_model(f"{PY}@step1000@fp16") == PY
    with pytest.raises(ValueError):
        modelcfg.model_name(PY, None, "int8")


def test_bf16_refused_for_pythia_only():
    with pytest.raises(SystemExit, match="bfloat16 is not allowed for Pythia"):
        modelcfg.check_dtype("EleutherAI/pythia-1b", "bfloat16")
    modelcfg.check_dtype("EleutherAI/pythia-1b", "float16")
    modelcfg.check_dtype(QWEN, "bfloat16")


def test_load_models_new_and_flat_forms(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(CFG))
    assert modelcfg.load_models(p)[P28] == {"corpus": "pile", "role": "sensitivity", "dtype": "float16"}
    p.write_text(json.dumps({PY: "pile", "reference": QWEN}))
    m = modelcfg.load_models(p)
    assert m[PY] == {"corpus": "pile", "role": "primary", "dtype": "float32"}
    assert m[QWEN]["role"] == "reference"
    for bad in ({PY: {"corpus": "pile", "role": "main"}}, {PY: {"corpus": "pile", "role": "primary", "dtype": "int8"}},
                {PY: 3}, {PY: {"corpus": None, "role": "reference"}, QWEN: {"corpus": None, "role": "reference"}}):
        p.write_text(json.dumps(bad))
        with pytest.raises(SystemExit):
            modelcfg.load_models(p)


def test_modelcfg_cli_lists_reference_last(tmp_path, capsys):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({QWEN: CFG[QWEN], PY: CFG[PY], P28: CFG[P28]}))
    modelcfg.main(["modelcfg.py", "list", str(p)])
    assert capsys.readouterr().out.splitlines() == [f"{PY} float32", f"{P28} float16", f"{QWEN} float32"]
    modelcfg.main(["modelcfg.py", "reference", str(p)])
    assert capsys.readouterr().out.strip() == QWEN


def test_project_models_json_follows_decisions():
    m = modelcfg.load_models(os.path.join(ROOT, "models.json"))
    assert m[P28] == {"corpus": "pile", "role": "sensitivity", "dtype": "float16"}
    assert m[OLMO] == {"corpus": "dolma", "role": "primary", "dtype": "float32"}
    assert m[QWEN]["role"] == "reference"
    assert "allenai/OLMo-1B-hf" not in m
    assert all(c["dtype"] == "float32" for k, c in m.items() if k != P28)
    assert [k for k, c in m.items() if c["role"] == "reference"] == [QWEN]


# ------------------------------------------------------------------ analyze
def test_corpus_role_and_ref_lookup_with_suffix():
    analyze.MODEL_CFG.update(CFG)
    analyze.MODEL_CORPUS.update({k: c["corpus"] for k, c in CFG.items() if c["role"] != "reference"})
    assert analyze.corpus_of(f"{OLMO}@fp32") == "dolma"
    assert analyze.corpus_of(f"{PY}@step1000@fp32") == "pile"
    assert analyze.role_of(f"{P28}@fp16") == "sensitivity"
    assert analyze.role_of(f"{QWEN}@fp32") == "reference"
    assert analyze.resolve_model(ALL, QWEN) == f"{QWEN}@fp32"
    assert analyze.resolve_model(ALL, f"{QWEN}@fp32") == f"{QWEN}@fp32"
    with pytest.raises(SystemExit, match="matches several dtypes"):
        analyze.resolve_model(ALL + [f"{QWEN}@fp16"], QWEN)


def test_names_without_dtype_rejected(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="without a dtype suffix"):
        run_analyze(tmp_path, monkeypatch, [PY, f"{OLMO}@fp32"], cfg=None, ref=None)


def test_different_dtypes_never_pooled(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="different dtypes for the same model.*pythia-410m@fp16, .*pythia-410m@fp32"):
        run_analyze(tmp_path / "a", monkeypatch, [f"{PY}@fp32", f"{PY}@fp16"], cfg=None, ref=None)
    # with models.json, a dtype other than the listed one is refused outright
    with pytest.raises(SystemExit, match="computed in fp16, but models.json lists"):
        run_analyze(tmp_path / "b", monkeypatch, [f"{PY}@fp32", f"{PY}@fp16"])
    # pooling check is applied to likelihood rows too
    with pytest.raises(SystemExit, match="different dtypes"):
        analyze.check_no_pooling(pd.DataFrame({"model": [f"{PY}@fp32", f"{PY}@fp16"]}))


def test_sensitivity_excluded_by_default(tmp_path, monkeypatch, capsys):
    run_analyze(tmp_path, monkeypatch, ALL)
    assert "sensitivity models ['EleutherAI/pythia-2.8b@fp16'] excluded" in capsys.readouterr().out
    for f in ("strata_accuracy.csv", "regression.csv", "loglik_strata.csv", "strata_n.csv"):
        assert set(pd.read_csv(tmp_path / f).model) == {f"{PY}@fp32", f"{OLMO}@fp32"}, f
    assert not list(tmp_path.glob("*_sensitivity.csv"))


def test_include_sensitivity_writes_separate_files(tmp_path, monkeypatch):
    run_analyze(tmp_path / "main", monkeypatch, ALL)
    run_analyze(tmp_path / "both", monkeypatch, ALL, "--include-sensitivity")
    for f in ("strata_accuracy", "sensitivity", "regression", "loglik_strata", "loglik_regression", "strata_n"):
        a = pd.read_csv(tmp_path / "main" / f"{f}.csv")
        b = pd.read_csv(tmp_path / "both" / f"{f}.csv")
        pd.testing.assert_frame_equal(a, b)  # main outputs unchanged by the flag
        s = pd.read_csv(tmp_path / "both" / f"{f}_sensitivity.csv")
        assert set(s.model) == {f"{P28}@fp16"}, f


def test_ref_model_given_with_or_without_suffix(tmp_path, monkeypatch):
    run_analyze(tmp_path / "bare", monkeypatch, ALL, ref=QWEN)
    run_analyze(tmp_path / "full", monkeypatch, ALL, ref=f"{QWEN}@fp32")
    for f in ("regression.csv", "loglik_regression.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "bare" / f), pd.read_csv(tmp_path / "full" / f))
    assert (pd.read_csv(tmp_path / "bare" / "loglik_regression.csv").controls.str.contains("ref_")).all()


def test_one_stratum_pair_is_reported_not_dropped(tmp_path, monkeypatch, capsys):
    run_analyze(tmp_path, monkeypatch, ALL)
    out = capsys.readouterr().out
    reg = pd.read_csv(tmp_path / "regression.csv")
    row = reg[(reg.model == f"{PY}@fp32") & (reg.benchmark == "gsm8k")]
    assert len(row) == 1
    assert row.skipped_reason.iloc[0] == "one stratum only: no contaminated items (overlap >= thr)"
    assert np.isnan(row.odds_ratio_contaminated.iloc[0])
    assert f"warning: regression skipped for {PY}@fp32 / gsm8k: one stratum only" in out
    assert re.search(r"gsm8k x pile: clean n=89, partial n=1, contaminated n=0; report as a clean-only "
                     r"negative control for EleutherAI/pythia-410m@fp32", out)
    fitted = reg[reg.skipped_reason.isna()]
    assert len(fitted) == 3 and fitted.odds_ratio_contaminated.notna().all()


def test_regression_skip_reasons():
    d = pd.DataFrame({"contaminated": [1, 1], "correct": [0, 1]})
    assert analyze.regression_skip_reason(d) == "one stratum only: every item contaminated"
    d = pd.DataFrame({"contaminated": [0, 1], "correct": [0, 0]})
    assert analyze.regression_skip_reason(d) == "constant outcome: every item incorrect"
    assert analyze.regression_skip_reason(d.iloc[0:0]).startswith("no rows")
    assert analyze.regression_skip_reason(pd.DataFrame({"contaminated": [0, 1], "correct": [0, 1]})) is None


# ------------------------------------------------------------------ report
def test_report_keeps_suffix_and_separates_sensitivity(tmp_path, monkeypatch):
    run_analyze(tmp_path, monkeypatch, ALL, "--include-sensitivity")
    monkeypatch.setattr(sys, "argv", ["report.py", "--in-dir", ".", "--out-dir", "po"])
    report.main()
    tex = (tmp_path / "po" / "tab_regression.tex").read_text(encoding="utf-8")
    assert "pythia-410m@fp32" in tex and "one stratum only" in tex and "pythia-2.8b" not in tex
    monkeypatch.setattr(sys, "argv", ["report.py", "--in-dir", ".", "--suffix", "_sensitivity", "--out-dir", "ps"])
    report.main()
    tex = (tmp_path / "ps" / "tab_strata_accuracy.tex").read_text(encoding="utf-8")
    assert "Sensitivity models only." in tex and "pythia-2.8b@fp16" in tex and "pythia-410m" not in tex
    assert r"\label{tab:strata_accuracy_sensitivity}" in tex


# ------------------------------------------------------------------ evaluate / itemloglik
def run_script(script, *args, cwd, ok=True):
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-B", os.path.join(ROOT, script), *args], cwd=cwd, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    if ok:
        assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-3000:]
    return p


@pytest.mark.parametrize("script", ["evaluate.py", "itemloglik.py"])
def test_rows_carry_fp32_suffix_by_default(tmp_path, script):
    p = run_script(script, "--model", "EleutherAI/pythia-70m", "--benchmarks", "mmlu", "--limit", "2",
                   "--device", "cpu", "--out", "o.csv", cwd=tmp_path)
    assert "dtype torch.float32" in p.stdout
    assert set(pd.read_csv(tmp_path / "o.csv").model) == {"EleutherAI/pythia-70m@fp32"}


@pytest.mark.parametrize("script", ["evaluate.py", "itemloglik.py"])
def test_bf16_refused_for_pythia_before_loading(tmp_path, script):
    p = run_script(script, "--model", "EleutherAI/pythia-70m", "--benchmarks", "mmlu", "--limit", "2",
                   "--dtype", "bfloat16", "--out", "o.csv", cwd=tmp_path, ok=False)
    assert p.returncode != 0 and "bfloat16 is not allowed for Pythia" in p.stderr
    assert not (tmp_path / "o.csv").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_default_is_fp32_and_fp16_rows_are_kept_apart(tmp_path):
    common = ["--model", "EleutherAI/pythia-70m", "--benchmarks", "mmlu", "--limit", "3", "--device", "cuda",
              "--out", "ll.csv", "--skip-existing"]
    p = run_script("itemloglik.py", *common, cwd=tmp_path)
    assert "dtype torch.float32" in p.stdout  # no silent fp16 on GPU any more
    first = pd.read_csv(tmp_path / "ll.csv")
    run_script("itemloglik.py", *common, "--dtype", "float16", cwd=tmp_path)  # not skipped: other dtype
    both = pd.read_csv(tmp_path / "ll.csv")
    assert set(both.model) == {"EleutherAI/pythia-70m@fp32", "EleutherAI/pythia-70m@fp16"}
    assert not both.duplicated(["model", "benchmark", "id"]).any()
    pd.testing.assert_frame_equal(both[both.model.str.endswith("@fp32")].reset_index(drop=True), first)
    assert "nothing new to score" in run_script("itemloglik.py", *common, "--dtype", "float16", cwd=tmp_path).stdout


# validate_sample tests moved to tests/test_validate_protocol.py: the export protocol was replaced
# (overlap > 0 only, one snippet per item by one rule), see STATUS_FINAL.md.


# ------------------------------------------------------------------ .bat split
GPU_SCRIPTS = ["perturb_gsm8k.py", "perturb_humaneval.py", "evaluate.py", "itemloglik.py", "common.py",
               "execution.py", "modelcfg.py"]


def test_gpu_stage_scripts_never_read_overlap():
    for f in GPU_SCRIPTS:
        src = open(os.path.join(ROOT, f), encoding="utf-8").read()
        assert "overlap.csv" not in src and "overlap_cache" not in src and "import overlap" not in src, f
        assert not re.search(r"^\s*from overlap import", src, re.M), f
    bat = open(os.path.join(ROOT, "run_gpu_stages.bat"), encoding="utf-8").read().splitlines()
    code = [re.sub(r"echo[^)]*", "", ln, flags=re.I) for ln in bat if not ln.lstrip().upper().startswith("REM")]
    assert not [ln for ln in code if "overlap" in ln.lower()]  # only comments / messages mention it
    for f in GPU_SCRIPTS[:4]:
        assert f in "\n".join(bat)
    ana = open(os.path.join(ROOT, "run_analysis.bat"), encoding="utf-8").read()
    assert "analyze.py" in ana and "report.py" in ana and "evaluate.py" not in ana and "itemloglik.py" not in ana


@pytest.mark.skipif(sys.platform != "win32", reason="cmd.exe")
def test_gpu_stages_dry_run_uses_models_json_dtypes():
    env = dict(os.environ, HF_HUB_OFFLINE="1", CHECK_STORAGE_FREE_GB="1000")  # storage check: fixed, offline
    p = subprocess.run(["cmd", "/c", os.path.join(ROOT, "run_gpu_stages.bat"), "--dry-run"],
                       capture_output=True, text=True, timeout=120, env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    cmds = [ln[1:].strip() for ln in p.stdout.splitlines() if ln.startswith(">")]
    runs = [c for c in cmds if c.split()[2] in ("evaluate.py", "itemloglik.py")]
    assert len(runs) == 12 and all("--skip-existing" in c for c in runs)
    assert "bfloat16" not in p.stdout
    for c in runs:
        assert ("--dtype float16" in c) == ("pythia-2.8b" in c), c
    assert QWEN in runs[-1] and QWEN in runs[-2]
    assert "DRY RUN" in p.stdout
