"""regression.csv fix: convergence / separation detection, empty (not 0 / inf / 1) values for fits that
did not converge, Fisher exact columns, Holm/BH over converged fits only, report output, and
old-vs-new analyze.py on identical synthetic data."""
import csv
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

import analyze
import report
import texcheck
from test_dtype_roles_blinding import ALL, CFG, QWEN, loglik_df, overlap_df, results_df

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY = os.path.join(ROOT, "tests", "legacy", "analyze_before_regression_fix.py")
SEP, OK = "EleutherAI/pythia-sep@fp32", "EleutherAI/pythia-ok@fp32"


@pytest.fixture(autouse=True)
def clean_globals():
    analyze.MODEL_CORPUS.clear()
    analyze.MODEL_CFG.clear()
    yield
    analyze.MODEL_CORPUS.clear()
    analyze.MODEL_CFG.clear()


def separation_data(n=60):
    """Two Pile models on one benchmark. SEP: all 5..k contaminated items incorrect (separation).
    OK: mixed outcomes in every stratum (an ordinary converged fit)."""
    rng = np.random.default_rng(3)
    ov = pd.DataFrame([{"corpus": "pile", "benchmark": "mmlu", "id": i, "n_words": 10 + i % 9, "n_windows": 10,
                        "overlap": [0.0, 0.3, 0.8][i % 3], "max_count": i % 3} for i in range(n)])
    rows = []
    for m in (SEP, OK):
        for i in range(n):
            contaminated = i % 3 == 2
            c = 0 if (m == SEP and contaminated) else int(rng.random() < (0.35 if contaminated else 0.55))
            rows.append({"model": m, "benchmark": "mmlu", "id": i, "variant": "raw", "correct": c})
    return pd.DataFrame(rows), ov


def run_results(tmp_path, monkeypatch, res, ov):
    monkeypatch.chdir(tmp_path)
    analyze.analyze_results(res.copy(), ov, None, 0.5)
    return pd.read_csv("regression.csv"), list(csv.DictReader(open("regression.csv", encoding="utf-8")))


def test_perfect_separation_is_not_estimable(tmp_path, monkeypatch, capsys):
    res, ov = separation_data()
    reg, raw = run_results(tmp_path, monkeypatch, res, ov)
    r = reg.set_index("model").loc[SEP]
    assert r.converged == False  # noqa: E712
    assert r.skipped_reason == "not estimable: perfect separation: all 20 contaminated items incorrect"
    assert pd.isna(r.odds_ratio_contaminated) and pd.isna(r.or_ci_lo) and pd.isna(r.or_ci_hi) and pd.isna(r.p)
    assert pd.isna(r.p_holm) and pd.isna(r.p_bh)
    assert "not estimable: perfect separation" in capsys.readouterr().out


def test_empty_values_not_zero_inf_or_one(tmp_path, monkeypatch):
    res, ov = separation_data()
    _, raw = run_results(tmp_path, monkeypatch, res, ov)
    row = [r for r in raw if r["model"] == SEP][0]
    for c in ("odds_ratio_contaminated", "or_ci_lo", "or_ci_hi", "p", "p_holm", "p_bh"):
        assert row[c] == "", (c, row[c])  # empty in the CSV text, not 0.0 / inf / 1.0
    old_style = [r for r in raw if r["model"] == SEP and r["odds_ratio_contaminated"] in ("0.0", "inf")]
    assert old_style == []


def test_normal_case_converges_and_matches_plain_fit(tmp_path, monkeypatch):
    res, ov = separation_data()
    reg, _ = run_results(tmp_path, monkeypatch, res, ov)
    r = reg.set_index("model").loc[OK]
    assert r.converged == True and (pd.isna(r.skipped_reason) or r.skipped_reason == "")  # noqa: E712
    d = res[res.model == OK].merge(ov, on=["benchmark", "id"])
    d["contaminated"] = (d.overlap >= 0.5).astype(int)
    d["log_len"] = np.log(d.n_words.clip(lower=1))
    fit = smf.logit("correct ~ contaminated + log_len", data=d).fit(disp=0)
    assert r.odds_ratio_contaminated == pytest.approx(float(np.exp(fit.params["contaminated"])), rel=1e-12)
    assert r.p == pytest.approx(float(fit.pvalues["contaminated"]), rel=1e-12)
    assert (r.n_contaminated, r.n_clean) == (20, 20)
    # Holm / BH family: the converged fits only (here one fit, so adjusted p == p)
    assert r.p_holm == pytest.approx(r.p) and r.p_bh == pytest.approx(r.p)
    tab = [[int(d[d.contaminated == 1].correct.sum()), int((1 - d[d.contaminated == 1].correct).sum())],
           [int(d[d.overlap == 0].correct.sum()), int((1 - d[d.overlap == 0].correct).sum())]]
    assert r.fisher_p == pytest.approx(fisher_exact(tab)[1])


def test_multiplicity_family_is_converged_rows_only():
    reg = pd.DataFrame({"model": list("abcd"), "benchmark": ["x"] * 4, "n": [50] * 4,
                        "odds_ratio_contaminated": [2.0, 3.0, 1e-9, np.nan], "or_ci_lo": [1, 1, 0, np.nan],
                        "or_ci_hi": [4, 9, np.inf, np.nan], "p": [0.01, 0.04, 0.99, np.nan],
                        "skipped_reason": ["", "", "", "constant outcome: every item incorrect"]})
    extra = [{"converged": True, "_why": "", **{k: np.nan for k in ("n_contaminated", "n_clean", "fisher_or",
                                                                   "fisher_ci_lo", "fisher_ci_hi", "fisher_p")}}] * 2
    extra += [{**extra[0], "converged": False, "_why": "maximum likelihood did not converge"},
              {**extra[0], "converged": False, "_why": None}]
    out = analyze.finish_regression(reg.copy(), extra)
    np.testing.assert_allclose(out.p_holm[:2], multipletests([0.01, 0.04], method="holm")[1])
    np.testing.assert_allclose(out.p_bh[:2], multipletests([0.01, 0.04], method="fdr_bh")[1])
    assert out.p_holm[2:].isna().all() and out.p[2:].isna().all()
    assert out.skipped_reason[2] == "not estimable: maximum likelihood did not converge"
    assert out.skipped_reason[3] == "constant outcome: every item incorrect"  # existing reason kept
    assert list(out.columns)[:len(reg.columns)] == list(reg.columns)  # existing columns first, in their order


@pytest.mark.parametrize("table,exp", [
    # Tea-tasting table. OR and p as printed by R's fisher.test. CI: exact bounds solved independently
    # from the noncentral hypergeometric (weights 1,16,36,16,1; P(X>=3)=0.025 and P(X<=3)=0.025 by brentq);
    # R's printed CI (0.2117329, 621.9337505) differs only within uniroot's default tolerance.
    ([[3, 1], [1, 3]], {"or": 6.408309, "lo": 0.2117355954, "hi": 626.2435305888, "p": 0.4857143}),
    # hand-computed: hypergeometric P(X=0)=35/220, P(X=3)=10/220 -> two-sided p = 45/220; OR 0
    ([[0, 5], [3, 4]], {"or": 0.0, "lo": 0.0, "hi": None, "p": 45 / 220}),
])
def test_fisher_against_known_values(table, exp):
    (a, b), (c, d) = table
    rows = ([{"contaminated": 1, "overlap": 0.9, "correct": 1}] * a + [{"contaminated": 1, "overlap": 0.9, "correct": 0}] * b
            + [{"contaminated": 0, "overlap": 0.0, "correct": 1}] * c + [{"contaminated": 0, "overlap": 0.0, "correct": 0}] * d
            + [{"contaminated": 0, "overlap": 0.2, "correct": 1}] * 3)  # partial items: not in the 2x2
    f = analyze.fisher_columns(pd.DataFrame(rows))
    assert (f["n_contaminated"], f["n_clean"]) == (a + b, c + d)
    assert f["fisher_p"] == pytest.approx(exp["p"], rel=1e-6)
    assert f["fisher_or"] == pytest.approx(exp["or"], rel=1e-5, abs=1e-9)
    assert f["fisher_ci_lo"] == pytest.approx(exp["lo"], rel=1e-8, abs=1e-9)
    if exp["hi"] is not None:
        assert f["fisher_ci_hi"] == pytest.approx(exp["hi"], rel=1e-8)
    else:
        assert np.isfinite(f["fisher_ci_hi"]) and f["fisher_ci_hi"] > 0


def test_fisher_empty_when_a_group_is_missing():
    f = analyze.fisher_columns(pd.DataFrame({"contaminated": [0, 0], "overlap": [0.0, 0.0], "correct": [1, 0]}))
    assert f["n_contaminated"] == 0 and np.isnan(f["fisher_or"]) and np.isnan(f["fisher_p"])


def test_report_table_and_forest(tmp_path, monkeypatch):
    res, ov = separation_data()
    run_results(tmp_path, monkeypatch, res, ov)
    pts = {}

    def fake_save(fig, out_dir, name):
        ax = fig.axes[0]
        pts[name] = (sum(len(l.get_xdata()) for l in ax.lines if l.get_linestyle() == "None"),
                     [t.get_text() for t in ax.texts])
        report.plt.close(fig)

    reg = pd.read_csv("regression.csv")
    monkeypatch.setattr(report, "save", fake_save)
    report.fig_forest(reg, ".")
    assert pts["fig_odds_ratio_forest"] == (1, ["not estimable (separation / no convergence)"])
    monkeypatch.undo()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["report.py", "--in-dir", ".", "--out-dir", "po"])
    report.main()
    tex = open(tmp_path / "po" / "tab_regression.tex", encoding="utf-8").read()
    sep_row = [ln for ln in tex.splitlines() if "pythia-sep@fp32" in ln][0]
    assert "not estimable" in sep_row and "perfect separation" in sep_row
    cells = [c.strip() for c in sep_row.rstrip("\\ ").split("&")]
    # logistic columns: OR cell says "not estimable"; OR low/high, p, p Holm are empty ("--"), never 0.00 / inf / 1.000
    assert cells[3] == "not estimable" and cells[4:8] == ["--", "--", "--", "--"]
    assert cells[8] == "0.00" and cells[11] not in ("--", "")  # Fisher OR 0 (all contaminated wrong), Fisher p
    assert "Fisher OR" in tex and texcheck.check_tex(tex) == []


def run_script(script, d, ref=QWEN, cfg=True, extra=()):
    argv = [sys.executable, "-B", script, "--results", "results.csv", "--overlap", "overlap.csv",
            "--loglik", "itemloglik.csv", *(["--ref-model", ref] if ref else []),
            *(["--model-corpus-map", "models.json"] if cfg else []), *extra]
    p = subprocess.run(argv, cwd=d, capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, PYTHONPATH=ROOT, RUN_MANIFEST=str(d / "m.jsonl")))
    assert p.returncode == 0, p.stderr[-2000:]


def compare(old, new):
    diffs = {}
    for f in sorted(glob.glob(str(old / "*.csv"))):
        name = os.path.basename(f)
        if name in ("results.csv", "itemloglik.csv", "overlap.csv"):
            continue
        a, b = pd.read_csv(f), pd.read_csv(new / name)
        try:
            pd.testing.assert_frame_equal(a, b[list(a.columns)])
        except AssertionError:
            diffs[name] = (a, b)
    return diffs


def test_old_vs_new_identical_without_separation(tmp_path):
    for tag in ("old", "new"):
        d = tmp_path / tag
        d.mkdir()
        results_df(ALL).to_csv(d / "results.csv", index=False)
        loglik_df(ALL).to_csv(d / "itemloglik.csv", index=False)
        overlap_df().to_csv(d / "overlap.csv", index=False)
        (d / "models.json").write_text(json.dumps(CFG))
        run_script(LEGACY if tag == "old" else os.path.join(ROOT, "analyze.py"), d, extra=["--include-sensitivity"])
    assert compare(tmp_path / "old", tmp_path / "new") == {}  # every pre-existing column of every file identical
    reg = pd.read_csv(tmp_path / "new" / "regression.csv")
    n = len(analyze.REG_DIAG_COLS)
    assert list(reg.columns)[-n - 1:-1] == analyze.REG_DIAG_COLS  # appended, followed by
    assert list(reg.columns)[-1] == "controls"  # the controls column (appended 2026-11-14)
    assert (reg.controls == "contaminated + log_len + ref_correct").all()  # reference has results rows here


def test_old_vs_new_with_separation_differs_only_where_requested(tmp_path):
    res, ov = separation_data()
    for tag in ("old", "new"):
        d = tmp_path / tag
        d.mkdir()
        res.to_csv(d / "results.csv", index=False)
        ov.to_csv(d / "overlap.csv", index=False)
        pd.DataFrame([{"model": m, "benchmark": "mmlu", "id": i, "ll_question": -1.0 - (i % 5) * 0.1 - (m == SEP),
                       "ll_question_answer": -2.0 + (i % 7) * 0.05, "mink_question": -3.0 - (i % 4) * 0.2,
                       "mink_question_answer": -4.0 + (i % 3) * 0.1} for m in (SEP, OK) for i in range(60)]
                     ).to_csv(d / "itemloglik.csv", index=False)
        run_script(LEGACY if tag == "old" else os.path.join(ROOT, "analyze.py"), d, ref=None, cfg=False)
    diffs = compare(tmp_path / "old", tmp_path / "new")
    assert set(diffs) == {"regression.csv"}  # every other output file identical
    a, b = diffs["regression.csv"]
    a, b = a.set_index("model"), b.set_index("model")
    pd.testing.assert_series_equal(a.loc[OK], b.loc[OK][a.columns], check_names=False)  # converged row identical
    changed = [c for c in a.columns if not (pd.isna(a.loc[SEP, c]) and pd.isna(b.loc[SEP, c]))
               and a.loc[SEP, c] != b.loc[SEP, c]]
    assert sorted(changed) == sorted(["odds_ratio_contaminated", "or_ci_lo", "or_ci_hi", "p", "skipped_reason"])
    assert a.loc[SEP, "or_ci_hi"] == np.inf or a.loc[SEP, "odds_ratio_contaminated"] < 1e-3  # old: fake estimate
    assert b.loc[SEP, ["odds_ratio_contaminated", "or_ci_lo", "or_ci_hi", "p"]].isna().all()
