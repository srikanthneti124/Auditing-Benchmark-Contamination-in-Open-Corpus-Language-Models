"""check_perturbed_sample.py: export and score modes on fixtures built with the real perturbation code."""
import json
import os
import sys

import pandas as pd
import pytest

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
import check_perturbed_sample as cps  # noqa: E402
import perturb_gsm8k  # noqa: E402
import perturb_humaneval  # noqa: E402
from common import load_benchmark  # noqa: E402
from validate_sample import wilson  # noqa: E402


@pytest.fixture(scope="module")
def gsm8k_recs():
    recs = [r for r, _ in (perturb_gsm8k.try_perturb(it, 0) for it in load_benchmark("gsm8k", 60, 0)) if r]
    assert len(recs) >= 12
    return recs


@pytest.fixture(scope="module")
def he_recs():
    recs = [r for r, _ in (perturb_humaneval.try_perturb(it, 0) for it in load_benchmark("humaneval", 8, 0)) if r]
    assert len(recs) >= 4
    return recs


def write_jsonl(path, recs):
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["check_perturbed_sample.py", *argv])
    cps.main()


def test_export_gsm8k_sheet(tmp_path, monkeypatch, gsm8k_recs):
    monkeypatch.chdir(tmp_path)
    write_jsonl("p.jsonl", gsm8k_recs)
    run(monkeypatch, "export-gsm8k", "--perturbed", "p.jsonl", "--n", "10", "--out", "a.csv")
    df = pd.read_csv("a.csv", keep_default_na=False, encoding="utf-8-sig")
    assert list(df.columns) == cps.GSM8K_COLS and len(df) == 10
    by_id = {r["id"]: r for r in gsm8k_recs}
    for r in df.itertuples():
        src = by_id[r.id]
        assert (r.original_question, r.perturbed_question) == (src["original_question"], src["perturbed_question"])
        assert str(r.perturbed_answer) == src["perturbed_answer"]
        assert json.loads(r.number_mapping) == src["number_mapping"]
    assert (df.verdict == "").all() and (df.notes == "").all()
    run(monkeypatch, "export-gsm8k", "--perturbed", "p.jsonl", "--n", "10", "--out", "b.csv")
    assert list(pd.read_csv("b.csv").id) == list(df.id)  # seeded
    with pytest.raises(SystemExit, match="already exists"):
        run(monkeypatch, "export-gsm8k", "--perturbed", "p.jsonl", "--out", "a.csv")
    run(monkeypatch, "export-gsm8k", "--perturbed", "p.jsonl", "--n", "50", "--out", "c.csv")
    assert len(pd.read_csv("c.csv")) == min(50, len(gsm8k_recs))  # default size 50, capped by availability


def test_score_fraction_wilson_and_wrong_list(tmp_path, monkeypatch, capsys, gsm8k_recs):
    monkeypatch.chdir(tmp_path)
    write_jsonl("p.jsonl", gsm8k_recs)
    run(monkeypatch, "export-gsm8k", "--perturbed", "p.jsonl", "--n", "10", "--out", "a.csv")
    df = pd.read_csv("a.csv", keep_default_na=False, encoding="utf-8-sig")
    df["verdict"] = ["correct"] * 6 + ["wrong"] * 2 + ["unclear", ""]
    df.loc[6, "notes"] = "answer recomputed wrongly"
    df.to_csv("filled.csv", index=False)
    capsys.readouterr()
    run(monkeypatch, "score", "--filled", "filled.csv")
    out = capsys.readouterr().out
    s = pd.read_csv("filled_score.csv").iloc[0]
    assert (s.correct, s.wrong, s.unclear_excluded, s.unlabelled) == (6, 2, 1, 1)
    assert s.fraction_correct == pytest.approx(0.75)
    assert (s.wilson_lo, s.wilson_hi) == pytest.approx(wilson(6, 8))
    assert f"WRONG id {df.id[6]}: answer recomputed wrongly" in out and f"WRONG id {df.id[7]}" in out
    df.loc[0, "verdict"] = "maybe"
    df.to_csv("bad.csv", index=False)
    with pytest.raises(SystemExit, match="invalid verdicts"):
        run(monkeypatch, "score", "--filled", "bad.csv")


def test_export_humaneval_runs_renamed_solution_on_original_tests(tmp_path, monkeypatch, capsys, he_recs):
    monkeypatch.chdir(tmp_path)
    broken = dict(he_recs[0], perturbed_canonical_solution="    return None\n")
    write_jsonl("p.jsonl", [broken] + he_recs[1:])
    run(monkeypatch, "export-humaneval", "--perturbed", "p.jsonl", "--n", "30", "--out", "h.csv")
    out = capsys.readouterr().out
    df = pd.read_csv("h.csv", keep_default_na=False, encoding="utf-8-sig")
    assert list(df.columns) == cps.HE_COLS and len(df) == len(he_recs)
    ok = df.set_index("id").canonical_passes_original_tests.astype(str)
    assert ok[broken["id"]] == "False"
    assert (ok.drop(broken["id"]) == "True").all()
    n = len(he_recs)
    assert f"renamed canonical solution vs ORIGINAL tests: {n - 1}/{n} pass" in out
    assert f"FAIL {broken['task_id']}" in out
    row = df[df.id == he_recs[1]["id"]].iloc[0]
    assert row.original_prompt == he_recs[1]["original_prompt"] and row.perturbed_prompt == he_recs[1]["perturbed_prompt"]
    df["verdict"] = "correct"
    df.to_csv("hf.csv", index=False)
    capsys.readouterr()
    run(monkeypatch, "score", "--filled", "hf.csv")
    assert f"passes original tests: {n - 1}/{n}" in capsys.readouterr().out


def test_missing_input_is_reported(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="not found"):
        run(monkeypatch, "export-gsm8k", "--perturbed", "none.jsonl")
