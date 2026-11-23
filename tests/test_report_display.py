"""report.py display fixes: non-estimable odds ratios are labelled (not drawn) in the forest plot, and the
likelihood figure title / table caption name only the controls actually used. Statistics are untouched."""
import numpy as np
import pandas as pd
import pytest

import report


@pytest.fixture
def captured(monkeypatch):
    figs = {}

    def fake_save(fig, out_dir, name):
        ax = fig.axes[0]
        figs[name] = {"xlim": ax.get_xlim(), "texts": [t.get_text() for t in ax.texts],
                      "suptitle": fig._suptitle.get_text() if fig._suptitle else "",
                      "n_points": sum(len(l.get_xdata()) for l in ax.lines if l.get_linestyle() == "None")}
        report.plt.close(fig)

    monkeypatch.setattr(report, "save", fake_save)
    return figs


def test_forest_labels_separated_fit_instead_of_drawing_it(captured):
    reg = pd.DataFrame([
        {"model": "o/a@fp32", "benchmark": "mmlu", "n": 17, "odds_ratio_contaminated": 2.5e-10, "or_ci_lo": 0.0,
         "or_ci_hi": np.inf, "p": 0.9998},
        {"model": "o/b@fp32", "benchmark": "mmlu", "n": 17, "odds_ratio_contaminated": 0.27, "or_ci_lo": 0.02,
         "or_ci_hi": 3.69, "p": 0.33}])
    report.fig_forest(reg, ".")
    f = captured["fig_odds_ratio_forest"]
    assert f["texts"] == ["not estimable (separation / no convergence)"]
    assert f["n_points"] == 1
    assert f["xlim"][0] == pytest.approx(0.02 / 1.5) and f["xlim"][1] == pytest.approx(3.69 * 1.5)
    assert reg.odds_ratio_contaminated.iloc[0] == 2.5e-10  # input untouched


def test_forest_all_estimable_has_no_label(captured):
    reg = pd.DataFrame([{"model": "o/b@fp32", "benchmark": "mmlu", "n": 50, "odds_ratio_contaminated": 2.0,
                         "or_ci_lo": 1.1, "or_ci_hi": 3.6, "p": 0.02}])
    report.fig_forest(reg, ".")
    assert captured["fig_odds_ratio_forest"]["texts"] == [] and captured["fig_odds_ratio_forest"]["n_points"] == 1


@pytest.mark.parametrize("controls,phrase", [
    (["contaminated + log_len", "contaminated + log_len"], "adjusted for length (95% CI)"),
    (["contaminated + log_len + ref_ll_question"] * 2, "adjusted for length and reference model (95% CI)"),
    (["contaminated + log_len + ref_ll_question", "contaminated + log_len"],
     "adjusted for length (and reference model where available)"),
])
def test_loglik_title_names_actual_controls(captured, controls, phrase):
    llreg = pd.DataFrame({"model": ["o/a@fp32", "o/b@fp32"], "benchmark": ["gsm8k", "gsm8k"],
                          "score": ["ll_question"] * 2, "n": [20, 20], "controls": controls,
                          "coef_contaminated": [0.1, 0.2], "ci_lo": [0.0, 0.1], "ci_hi": [0.2, 0.3], "p": [0.1, 0.01]})
    report.fig_loglik(llreg, ".")
    assert phrase in captured["fig_loglik_effect"]["suptitle"]
