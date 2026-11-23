"""Additive statistics and reporting: p_holm / p_bh, primary_outcome, coverage.csv, figure titles,
and a structural LaTeX check (pdflatex is not installed; compilation is UNVERIFIED)."""
import glob
import sys

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

import analyze
import report
import texcheck
from test_dtype_roles_blinding import ALL, N, P28, PY, run_analyze

ACC_COLS = ["model", "benchmark", "stratum", "n", "acc", "ci_lo", "ci_hi"]
LLREG_COLS = ["model", "benchmark", "score", "n", "controls", "coef_contaminated", "ci_lo", "ci_hi", "p",
              "higher_in_contaminated_p05", "error"]


@pytest.fixture(autouse=True)
def clean_globals():
    analyze.MODEL_CORPUS.clear()
    analyze.MODEL_CFG.clear()
    yield
    analyze.MODEL_CORPUS.clear()
    analyze.MODEL_CFG.clear()


def test_holm_and_bh_columns_are_appended(tmp_path, monkeypatch, capsys):
    run_analyze(tmp_path, monkeypatch, ALL, "--include-sensitivity")
    for f in ("loglik_regression.csv", "loglik_regression_sensitivity.csv"):
        rg = pd.read_csv(tmp_path / f)
        assert list(rg.columns)[-2:] == ["p_holm", "p_bh"]
        assert set(LLREG_COLS) <= set(rg.columns)
        m = rg.p.notna()
        assert m.sum() >= 2
        np.testing.assert_allclose(rg.p_holm[m], multipletests(rg.p[m], method="holm")[1])
        np.testing.assert_allclose(rg.p_bh[m], multipletests(rg.p[m], method="fdr_bh")[1])
        assert rg.p_holm[~m].isna().all() and rg.p_bh[~m].isna().all()
        assert (rg.p_holm[m] >= rg.p[m] - 1e-12).all() and (rg.p_bh[m] >= rg.p[m] - 1e-12).all()
    assert "family = all" in capsys.readouterr().out


def test_add_multiplicity_without_p():
    rg = analyze.add_multiplicity(pd.DataFrame({"model": ["m"], "error": ["x"]}))
    assert rg.p_holm.isna().all() and rg.p_bh.isna().all()


def test_primary_outcome_rule():
    acc = pd.DataFrame([
        {"model": "a", "benchmark": "gsm8k", "stratum": "all", "n": 100, "acc": 0.01, "ci_lo": 0.0, "ci_hi": 0.04},
        {"model": "a", "benchmark": "gsm8k", "stratum": "clean", "n": 90, "acc": 0.01, "ci_lo": 0.0, "ci_hi": 0.05},
        {"model": "a", "benchmark": "mmlu", "stratum": "all", "n": 100, "acc": 0.26, "ci_lo": 0.2, "ci_hi": 0.31},
        {"model": "b", "benchmark": "mmlu", "stratum": "all", "n": 100, "acc": 0.45, "ci_lo": 0.4, "ci_hi": 0.5},
        {"model": "b", "benchmark": "new", "stratum": "all", "n": 100, "acc": 0.0, "ci_lo": 0.0, "ci_hi": 0.0},
    ])
    out = analyze.add_primary_outcome(acc, 0.05)
    assert list(out.columns) == ACC_COLS + ["primary_outcome"]
    assert list(out.primary_outcome) == ["likelihood", "likelihood", "accuracy", "accuracy", "accuracy"]
    pd.testing.assert_frame_equal(out[ACC_COLS], acc)  # existing columns untouched


def test_primary_outcome_matches_report_rule(tmp_path, monkeypatch):
    run_analyze(tmp_path, monkeypatch, ALL)
    acc = pd.read_csv(tmp_path / "strata_accuracy.csv")
    assert list(acc.columns) == ACC_COLS + ["primary_outcome"]
    sup = report.outcome_support(acc, None, 0.05)
    mine = {(m, b): o for m, b, o in zip(acc.model, acc.benchmark, acc.primary_outcome)}
    for r in sup.itertuples():
        assert (mine[(r.model, r.benchmark)] == "likelihood") == (r.near_chance == "yes")
    assert report.CHANCE is analyze.CHANCE


def test_coverage_table(tmp_path, monkeypatch):
    run_analyze(tmp_path, monkeypatch, ALL)
    cov = pd.read_csv(tmp_path / "coverage.csv").set_index("benchmark")
    assert list(cov.columns) == ["items_sampled", "items_overlap_computed_dolma", "items_overlap_computed_pile",
                                 "items_evaluated", "items_perturbable", "items_loglik",
                                 "models_accuracy", "models_loglik"]  # model counts appended 2026-11-14
    assert (cov.drop(columns=["models_accuracy", "models_loglik"]) == N).all().all()


def test_coverage_counts_losses():
    ov_all = pd.DataFrame({"corpus": ["pile"] * 4 + ["dolma"] * 4, "benchmark": ["b"] * 8, "id": [1, 2, 3, 4] * 2,
                           "overlap": [0.1, np.nan, 0.0, 0.5, 0.0, 0.0, np.nan, np.nan]})
    ov = ov_all[ov_all.corpus == "pile"]  # dolma excluded as incomplete
    res = pd.DataFrame({"model": ["m"] * 5, "benchmark": ["b"] * 5, "id": [1, 2, 3, 1, 3],
                        "variant": ["raw", "raw", "raw", "perturbed", "perturbed"], "correct": [1] * 5})
    row = analyze.coverage(res, ov_all, ov).iloc[0]
    assert (row.items_sampled, row.items_overlap_computed_pile, row.items_overlap_computed_dolma,
            row.items_evaluated, row.items_perturbable) == (4, 3, 0, 3, 2)


def test_report_titles_follow_primary_outcome(tmp_path, monkeypatch):
    acc = pd.DataFrame([
        {"model": "org/near@fp32", "benchmark": "gsm8k", "stratum": s, "n": 50, "acc": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
         "primary_outcome": "likelihood"} for s in ("clean", "all")] + [
        {"model": "org/good@fp32", "benchmark": "gsm8k", "stratum": s, "n": 50, "acc": 0.4, "ci_lo": 0.3, "ci_hi": 0.5,
         "primary_outcome": "accuracy"} for s in ("clean", "all")] + [
        {"model": "org/good@fp32", "benchmark": "mmlu", "stratum": s, "n": 50, "acc": 0.5, "ci_lo": 0.4, "ci_hi": 0.6,
         "primary_outcome": "accuracy"} for s in ("clean", "all")])
    figs = {}

    def fake_save(fig, out_dir, name):
        figs[name] = [(ax.get_title(), [t.get_text() for t in ax.get_xticklabels() + ax.get_yticklabels()])
                      for ax in fig.axes]
        report.plt.close(fig)

    monkeypatch.setattr(report, "save", fake_save)
    report.dodge_plot(acc, "acc", "ci_lo", "ci_hi", report.STRATA_ORDER, "accuracy", str(tmp_path), "fig",
                      primary=report.primary_map(acc))
    (t_gsm, ticks_gsm), (t_mmlu, ticks_mmlu) = figs["fig"]
    assert t_gsm == "gsm8k\n" + report.NEAR_CHANCE_NOTE and t_mmlu == "mmlu"
    assert "near@fp32 †" in ticks_gsm and "good@fp32" in ticks_gsm and "good@fp32 †" not in ticks_gsm
    llreg = pd.DataFrame([{"model": "org/near@fp32", "benchmark": "gsm8k", "score": "ll_question", "n": 50,
                           "coef_contaminated": 0.1, "ci_lo": 0.0, "ci_hi": 0.2, "p": 0.04}])
    report.fig_loglik(llreg, str(tmp_path), report.primary_map(acc))
    assert any("[primary]" in t for _, ticks in figs["fig_loglik_effect"] for t in ticks)


def test_generated_tex_passes_structural_check(tmp_path, monkeypatch):
    run_analyze(tmp_path, monkeypatch, ALL, "--include-sensitivity")
    for suffix, out in (("", "po"), ("_sensitivity", "ps")):
        monkeypatch.setattr(sys, "argv", ["report.py", "--in-dir", ".", "--out-dir", out, "--suffix", suffix])
        report.main()
    files = glob.glob(str(tmp_path / "po" / "*.tex")) + glob.glob(str(tmp_path / "ps" / "*.tex"))
    names = {f.replace("\\", "/").rsplit("/", 1)[1] for f in files}
    assert {"tab_coverage.tex", "tab_loglik_regression.tex", "tab_regression.tex", "tab_strata_accuracy.tex"} <= names
    for f in files:
        assert texcheck.check_tex(open(f, encoding="utf-8").read()) == [], f
    assert texcheck.main(files) == 0
    tex = open(tmp_path / "po" / "tab_loglik_regression.tex", encoding="utf-8").read()
    assert "$p$ Holm" in tex and "$p$ BH" in tex


@pytest.mark.parametrize("bad,msg", [
    ("\\begin{table}\\begin{tabular}{ll}\na & b \\\\\n\\end{tabular}", "unclosed"),
    ("\\begin{tabular}{ll}\na & b & c \\\\\n\\end{tabular}", "3 cells, column spec has 2"),
    ("\\begin{tabular}{ll}\n50% & b \\\\\n\\end{tabular}", "unescaped '%'"),
    ("\\begin{tabular}{ll}\nmy_model & b \\\\\n\\end{tabular}", "unescaped '_'"),
    ("\\begin{tabular}{ll}\nissue #3 & b \\\\\n\\end{tabular}", "unescaped '#'"),
    ("\\begin{tabular}{ll}\n$x & b \\\\\n\\end{tabular}", "unbalanced $"),
    ("\\begin{table}\\caption{rate 5% here}\n\\end{table}", "caption: unescaped '%'"),
    ("\\begin{tabular}{ll}\n\\end{table}", "closes"),
])
def test_texcheck_detects_problems(bad, msg):
    assert any(msg in p for p in texcheck.check_tex(bad)), texcheck.check_tex(bad)


def test_texcheck_accepts_escapes_and_math():
    ok = ("\\begin{table}\n\\caption{Acc. (95\\% CI), $\\le$ chance}\n\\begin{tabular}{lrr}\n"
          "pythia\\_1b@fp32 & $<$0.001 & 50\\% \\\\\n\\textbackslash{} & a\\&b & \\#1 \\\\\n\\end{tabular}\n\\end{table}\n")
    assert texcheck.check_tex(ok) == []
