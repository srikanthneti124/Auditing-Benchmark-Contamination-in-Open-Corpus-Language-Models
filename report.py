"""Paper-ready figures and LaTeX tables from analyze.py outputs.

Figures (PDF + PNG, 300 dpi), one panel per benchmark:
  fig_accuracy_by_stratum   accuracy per model and stratum, bootstrap 95% CI
                            (CIs computed by analyze.py, read from the CSV)
  fig_raw_vs_adjusted       raw (all items) vs adjusted (clean subset) accuracy
  fig_sensitivity           raw - perturbed accuracy by stratum, bootstrap CI
  fig_odds_ratio_forest     odds ratio for `contaminated`, Wald 95% CI, log x
Tables (booktabs): one .tex per input CSV.

Palette: Okabe-Ito blue / orange / vermillion for clean / partial /
contaminated (validated for CVD separation), with distinct marker shapes so
identity never relies on colour alone.

Model names keep their dtype suffix (pythia-1.4b@fp32) in every figure and
table. Primary outcome (strata_accuracy.csv column primary_outcome, from
analyze.py): in accuracy figures a model x benchmark whose accuracy is near
chance is marked "†" and the panel title says accuracy is descriptive only;
in the likelihood figure such rows are marked "[primary]". coverage.csv ->
tab_coverage.tex; data_availability.csv -> tab_data_availability.tex.

Partial data (models with likelihood rows but no results.csv rows): every
accuracy figure gets a footnote and every table caption a sentence naming the
models with accuracy data and those with likelihood scores only; in the
likelihood figure such models are marked "[no accuracy]", and rows whose
coefficient could not be fitted are shown as "not estimable" instead of being
dropped; tab_outcome_support lists them with primary outcome "likelihood only
(no accuracy data)". The near-chance rule is unchanged. --suffix _sensitivity reads analyze.py's *_sensitivity.csv files
(role "sensitivity" models) so they are reported separately, never pooled.

Usage:
  python report.py --in-dir . --out-dir paper_outputs
  python report.py --in-dir . --suffix _sensitivity --out-dir paper_outputs/sensitivity
"""
import argparse
import os
import sys

import matplotlib
import matplotlib.ticker
import matplotlib.transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analyze  # noqa: E402
from manifest import write_manifest  # noqa: E402

STRATUM_STYLE = {
    "clean": ("#0072B2", "o"),
    "partial": ("#E69F00", "s"),
    "contaminated": ("#D55E00", "^"),
    "all": ("#4D4D4D", "D"),
}
INK, MUTED = "#222222", "#6B6B6B"
STRATA_ORDER = ["clean", "partial", "contaminated", "all"]

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def short(model):
    return model.split("/")[-1]


def primary_map(acc):
    if acc is None or "primary_outcome" not in acc:
        return {}
    return {(m, b): o for m, b, o in zip(acc.model, acc.benchmark, acc.primary_outcome)}


def mlabel(model, benchmark, primary):
    return short(model) + (" †" if primary.get((model, benchmark)) == "likelihood" else "")


NEAR_CHANCE_NOTE = "† accuracy near chance: descriptive only"
OR_DISPLAY_RANGE = (1e-3, 1e3)  # forest plot: odds ratios outside this range are labelled, not drawn


def footnote(fig, note):
    if note:
        fig.text(0.5, -0.02, plain(note), ha="center", va="top", fontsize=7, color=MUTED, wrap=True)


def save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {name}.pdf/.png")


def _panels(benchmarks, width_per=3.0, height=2.6):
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(width_per * len(benchmarks), height), squeeze=False)
    return fig, axes[0]


def dodge_plot(df, value, lo, hi, strata, ylabel, out_dir, name, zero_line=False, primary=None, note=""):
    primary = primary or {}
    benchmarks = sorted(df.benchmark.unique())
    fig, axes = _panels(benchmarks)
    for ax, b in zip(axes, benchmarks):
        g = df[df.benchmark == b]
        models = sorted(g.model.unique())
        present = [s for s in strata if s in set(g.stratum)]
        width = 0.7 / max(1, len(present))
        for j, s in enumerate(present):
            colour, marker = STRATUM_STYLE[s]
            gs = g[g.stratum == s].set_index("model").reindex(models)
            x = np.arange(len(models)) - 0.35 + width * (j + 0.5)
            y = gs[value].to_numpy(float)
            err = np.vstack([y - gs[lo].to_numpy(float), gs[hi].to_numpy(float) - y])
            ax.errorbar(x, y, yerr=np.where(np.isfinite(err), err, 0), fmt=marker, color=colour, ms=4.5,
                        elinewidth=1, capsize=0, label=s, mec="white", mew=0.6)
        if zero_line:
            ax.axhline(0, color=MUTED, lw=0.8, zorder=0)
        if value == "acc":  # accuracy cannot be negative; keep a readable range when all values are 0
            ax.set_ylim(-0.02, max(ax.get_ylim()[1], 0.1))
        ax.set_xticks(range(len(models)), [mlabel(m, b, primary) for m in models], rotation=30, ha="right")
        near = any(primary.get((m, b)) == "likelihood" for m in models)
        ax.set_title(f"{b}\n{NEAR_CHANCE_NOTE}" if near else b, fontsize=9 if not near else 8)
        ax.set_ylabel(ylabel if ax is axes[0] else "")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), title="stratum", title_fontsize=7)
    fig.tight_layout()
    footnote(fig, note)
    save(fig, out_dir, name)


def fig_raw_vs_adjusted(adj, acc, out_dir, note=""):
    primary = primary_map(acc)
    if acc is not None:
        ci = acc[acc.stratum.isin(["all", "clean"])].pivot_table(
            index=["benchmark", "model"], columns="stratum", values=["ci_lo", "ci_hi"])
        ci.columns = [f"{v}_{s}" for v, s in ci.columns]
        adj = adj.join(ci, on=["benchmark", "model"])
    benchmarks = sorted(adj.benchmark.unique())
    fig, axes = _panels(benchmarks, height=0.45 * adj.model.nunique() + 1.0)
    for ax, b in zip(axes, benchmarks):
        g = adj[adj.benchmark == b].sort_values("raw_acc").reset_index(drop=True)
        y = np.arange(len(g))
        ax.hlines(y, g.raw_acc, g.adjusted_acc, color="#BDBDBD", lw=1.5, zorder=1)
        for col, s, lab, dy in (("raw_acc", "all", "raw (all items with overlap)", 0.12), ("adjusted_acc", "clean", "adjusted (clean)", -0.12)):
            colour, marker = STRATUM_STYLE[s]
            if f"ci_lo_{s}" in g:
                ax.hlines(y + dy, g[f"ci_lo_{s}"], g[f"ci_hi_{s}"], color=colour, lw=1, zorder=2)
            ax.plot(g[col], y + dy, marker, color=colour, ms=4.5, label=lab, mec="white", mew=0.6, zorder=3)
        ax.set_yticks(y, [mlabel(m, b, primary) for m in g.model])
        ax.set_xlim(-0.02, max(ax.get_xlim()[1], 0.1))
        ax.set_xlabel("accuracy")
        near = any(primary.get((m, b)) == "likelihood" for m in g.model)
        ax.set_title(f"{b}\n{NEAR_CHANCE_NOTE}" if near else b, fontsize=9 if not near else 8)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    footnote(fig, note)
    save(fig, out_dir, "fig_raw_vs_adjusted")


def fig_forest(reg, out_dir, primary=None, note=""):
    primary = primary or {}
    if "odds_ratio_contaminated" not in reg:
        print("  skip forest plot: regression.csv has no odds ratios")
        return
    if "converged" in reg:  # analyze.py marks fits that did not converge; draw only converged ones
        conv = reg.converged.astype(str).str.lower().isin(["true", "1"])
        r = pd.concat([reg[conv].dropna(subset=["odds_ratio_contaminated"]),
                       reg[~conv & reg.skipped_reason.astype(str).str.startswith("not estimable")]]).copy()
        r["_conv"] = r.index.isin(reg[conv].index)
    else:
        r = reg.dropna(subset=["odds_ratio_contaminated"]).copy()
    if r.empty:
        print("  skip forest plot: no fitted regressions")
        return
    has_ci = {"or_ci_lo", "or_ci_hi"} <= set(r.columns)
    if not has_ci:
        print("  warning: regression.csv lacks or_ci_lo/or_ci_hi (re-run analyze.py); plotting points only")
    r = r.sort_values(["benchmark", "model"]).reset_index(drop=True)
    # Display only: a fit with an infinite / zero CI bound or an extreme OR (separation, no convergence)
    # is labelled "not estimable" instead of drawn, so it cannot stretch the log axis. Values in the
    # table and CSV are unchanged.
    if "_conv" in r:
        ok = r["_conv"].astype(bool) & r.odds_ratio_contaminated.notna()
    else:  # older regression.csv without the converged column: fall back to a range check
        ok = r.odds_ratio_contaminated.between(OR_DISPLAY_RANGE[0], OR_DISPLAY_RANGE[1])
        if has_ci:
            ok &= np.isfinite(r.or_ci_lo) & np.isfinite(r.or_ci_hi) & (r.or_ci_lo > 0)
    fig, ax = plt.subplots(figsize=(4.2, 0.28 * len(r) + 0.9))
    y = np.arange(len(r))[::-1]
    if has_ci:
        ax.hlines(y[ok], r.or_ci_lo[ok], r.or_ci_hi[ok], color=INK, lw=1)
    ax.plot(r.odds_ratio_contaminated[ok], y[ok], "s", color=INK, ms=4)
    ax.axvline(1, color=MUTED, lw=0.8, ls="--", zorder=0)
    ax.set_xscale("log")
    shown = r[ok] if ok.any() else r.iloc[0:0]
    lo = min(shown.odds_ratio_contaminated.min(), shown.or_ci_lo.min() if has_ci else np.inf) if len(shown) else 0.5
    hi = max(shown.odds_ratio_contaminated.max(), shown.or_ci_hi.max() if has_ci else 0) if len(shown) else 2
    ax.set_xlim(min(lo, 1) / 1.5, max(hi, 1) * 1.5)
    for yy in y[~ok.to_numpy()]:
        ax.text(1, yy, "not estimable (separation / no convergence)", ha="center", va="center", fontsize=6,
                color=MUTED, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ticks = [t for t in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100) if lo / 1.5 <= t <= hi * 1.5] or [1]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(y, [f"{b} · {mlabel(m, b, primary)}  (n={int(n)})" for b, m, n in zip(r.benchmark, r.model, r.n)])
    ax.set_xlabel("odds ratio, contaminated vs not (95% CI, log scale)")
    if any(primary.get((m, b)) == "likelihood" for m, b in zip(r.model, r.benchmark)):
        ax.set_title(NEAR_CHANCE_NOTE, fontsize=8)
    if "controls" in r and not r.controls.astype(str).str.contains("ref_").any():
        note = (note + " " if note else "") + "Controls: length only (no reference-model accuracy data)."
    fig.tight_layout()
    footnote(fig, note)
    save(fig, out_dir, "fig_odds_ratio_forest")


# ---------------------------------------------------------------- LaTeX
def tex_escape(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")):
        s = s.replace(a, b)
    return s


def regression_display(reg):
    """Display copy of regression.csv: 'not estimable' in the OR cell of fits that did not converge."""
    r = reg.copy()
    if "converged" in r and "odds_ratio_contaminated" in r:
        bad = ~r.converged.astype(str).str.lower().isin(["true", "1"])
        r["odds_ratio_contaminated"] = r.odds_ratio_contaminated.astype(object)
        r.loc[bad, "odds_ratio_contaminated"] = "not estimable"
    return r


def fmt_cell(col, v, d):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    if isinstance(v, str) and col != "model":
        return tex_escape(v)
    if isinstance(v, (float, np.floating)) and np.isinf(v):
        return r"$\infty$" if v > 0 else r"$-\infty$"
    if col == "fisher_p":
        return r"$<$0.001" if v < 0.001 else f"{v:.3f}"
    if col.startswith("fisher_"):
        return f"{v:.2f}"
    if col in ("p", "p_holm", "p_bh"):
        return r"$<$0.001" if v < 0.001 else f"{v:.3f}"
    if col in ("n",):
        return f"{int(v)}"
    if col.startswith("odds_ratio") or col.startswith("or_ci"):
        return f"{v:.2f}"
    if isinstance(v, (float, np.floating)):
        return f"{v:.{d}f}"
    if col == "model":
        return tex_escape(short(v))
    return tex_escape(v)


def write_table(df, cols, header, caption, label, path, d):
    cols = [c for c in cols if c in df.columns]
    align = "".join("r" if pd.api.types.is_numeric_dtype(df[c]) else "l" for c in cols)
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(header.get(c, tex_escape(c)) for c in cols) + r" \\", r"\midrule"]
    prev = None
    for _, row in df[cols].iterrows():
        if "benchmark" in cols and prev is not None and row["benchmark"] != prev:
            lines.append(r"\midrule")
        prev = row.get("benchmark")
        lines.append(" & ".join(fmt_cell(c, row[c], d) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  table  -> {os.path.basename(path)}")


def reg_controls_caption(reg):
    if "controls" not in reg:
        return ""
    c = reg.controls.dropna().astype(str)
    if len(c) and not c.str.contains("ref_").any():
        return "Controls: length only; the reference model has no accuracy data, so no reference-model control. "
    return "Controls: " + tex_escape("; ".join(sorted(set(c)))) + ". "


HEADER = {"model": "Model", "benchmark": "Benchmark", "stratum": "Stratum", "n": "$n$", "acc": "Acc.",
          "ci_lo": "CI low", "ci_hi": "CI high", "raw": "Raw", "perturbed": "Perturbed",
          "sensitivity": "Sensitivity", "odds_ratio_contaminated": "OR", "or_ci_lo": "OR low",
          "or_ci_hi": "OR high", "p": "$p$", "raw_acc": "Raw acc.", "adjusted_acc": "Adjusted acc.",
          "inflation": "Inflation", "score": "Score", "coef_contaminated": "$\\Delta$ log-prob",
          "chance": "Chance", "near_chance": "Near chance", "primary_outcome": "Primary outcome",
          "loglik_available": "Log-lik. data", "skipped_reason": "Not fitted", "p_holm": "$p$ Holm",
          "p_bh": "$p$ BH", "fisher_or": "Fisher OR", "fisher_ci_lo": "Fisher low", "fisher_ci_hi": "Fisher high",
          "fisher_p": "Fisher $p$","items_sampled": "Sampled", "items_evaluated": "Evaluated",
          "items_perturbable": "Perturbable", "items_loglik": "Log-lik. scored",
          "models_accuracy": "Models (acc.)", "models_loglik": "Models (log-lik.)", "role": "Role",
          "corpus": "Corpus", "accuracy": "Accuracy", "sensitivity_raw_minus_perturbed": "Raw$-$perturbed",
          "likelihood": "Log-lik.", "reference_control_accuracy": "Ref. control (acc.)",
          "reference_control_likelihood": "Ref. control (log-lik.)", "loglik_estimable": "Log-lik. estimable"}
CHANCE = analyze.CHANCE


def model_scope_note(acc, llreg):
    """One sentence naming which models have accuracy data and which have likelihood scores only."""
    acc_m = sorted(acc.model.unique()) if acc is not None else []
    ll_m = sorted(llreg.model.unique()) if llreg is not None else []
    only = [m for m in ll_m if m not in acc_m]
    if not only:
        return ""
    names = lambda ms: ", ".join(short(m) for m in ms)  # noqa: E731
    return (f"Accuracy, raw$-$perturbed sensitivity and the accuracy regression: {names(acc_m) or 'no model'} only; "
            f"{names(only)}: likelihood scores only.")


def plain(note):
    """Caption text -> figure text (matplotlib mathtext would mis-render the TeX minus)."""
    return note.replace("$-$", "-")


def outcome_support(acc, llreg, margin):
    """Which outcome each model x benchmark can support. Accuracy is treated as
    uninformative ("near chance") when the upper 95% CI bound of overall accuracy is
    <= chance + margin; then the likelihood analysis is the primary outcome.
    loglik_available = the model has likelihood rows for the benchmark; loglik_estimable =
    the contamination coefficient could be fitted. Models with likelihood rows but no
    accuracy rows are listed with primary_outcome 'likelihood only (no accuracy data)'."""
    a = acc[acc.stratum == "all"][["model", "benchmark", "acc", "ci_lo", "ci_hi"]].copy()
    a["chance"] = a.benchmark.map(CHANCE)
    nc = a.ci_hi <= a.chance + margin
    have, est = set(), set()
    if llreg is not None:
        have = set(zip(llreg.model, llreg.benchmark))
        if "coef_contaminated" in llreg:
            e = llreg.dropna(subset=["coef_contaminated"])
            est = set(zip(e.model, e.benchmark))
    ll_only = sorted(have - set(zip(a.model, a.benchmark)))
    a = pd.concat([a, pd.DataFrame(ll_only, columns=["model", "benchmark"])], ignore_index=True)
    a["chance"] = a.benchmark.map(CHANCE)
    a["has_acc"] = a.index < len(nc)
    nc = list(nc) + [None] * len(ll_only)
    a["loglik_available"] = [(m, b) in have for m, b in zip(a.model, a.benchmark)]
    a["loglik_estimable"] = [(m, b) in est for m, b in zip(a.model, a.benchmark)]

    def label(has_acc, near, ll, ok):
        lik = "likelihood (Min-K%, log-lik.)" if ok else "likelihood; contamination contrast not estimable"
        if not has_acc:
            return "likelihood only (no accuracy data)" + ("" if ok else "; contrast not estimable")
        if near:
            return lik if ll else "none: accuracy near chance, no likelihood data"
        return ("accuracy + likelihood" if ok else "accuracy (likelihood contrast not estimable)") if ll else "accuracy"
    a["primary_outcome"] = [label(h, n, l, o) for h, n, l, o in
                            zip(a.has_acc, nc, a.loglik_available, a.loglik_estimable)]
    a["near_chance"] = [("yes" if n else "no") if h else "n/a (no accuracy)" for h, n in zip(a.has_acc, nc)]
    a["loglik_available"] = a.loglik_available.map({True: "yes", False: "no"})
    a["loglik_estimable"] = a.loglik_estimable.map({True: "yes", False: "no"})
    cols = ["model", "benchmark", "acc", "ci_lo", "ci_hi", "chance", "near_chance", "loglik_available",
            "primary_outcome", "loglik_estimable"]  # original order; loglik_estimable appended
    return a[cols].sort_values(["benchmark", "model"])


def fig_loglik(llreg, out_dir, primary=None, acc_models=None):
    """Fitted coefficients are drawn; model x benchmark x score rows that could not be fitted (e.g. no
    contaminated item) are kept as labelled 'not estimable' rows, never dropped silently."""
    primary = primary or {}
    acc_models = set(acc_models or [])
    if "coef_contaminated" not in llreg or llreg.coef_contaminated.isna().all():
        print("  skip likelihood figure: no fitted likelihood regressions")
        return
    r = llreg
    scores = list(dict.fromkeys(r.score))
    fig, axes = plt.subplots(1, len(scores), figsize=(2.6 * len(scores) + 2.2, 0.28 * r.groupby("score").size().max() + 1.0),
                             squeeze=False, sharey=True)
    labels = None
    for ax, s in zip(axes[0], scores):
        g = r[r.score == s].sort_values(["benchmark", "model"]).reset_index(drop=True)
        y = np.arange(len(g))[::-1]
        ok = g.coef_contaminated.notna().to_numpy()
        ax.hlines(y[ok], g.ci_lo[ok], g.ci_hi[ok], color=INK, lw=1)
        ax.plot(g.coef_contaminated[ok], y[ok], "s", color=INK, ms=4)
        ax.axvline(0, color=MUTED, lw=0.8, ls="--", zorder=0)
        for yy in y[~ok]:
            ax.text(0.03, yy, "not estimable", ha="left", va="center", fontsize=6, color=MUTED,
                    transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData),
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
        ax.set_title(s.replace("_", " "))
        ax.set_xlabel("$\\Delta$ mean log-prob")
        labels = [f"{b} · {short(m)}" + (" [primary]" if primary.get((m, b)) == "likelihood" else
                                         " [no accuracy]" if acc_models and m not in acc_models else "")
                  for b, m in zip(g.benchmark, g.model)]
        ax.set_yticks(y, labels)
    note = ("; [primary] = likelihood is the primary outcome (accuracy near chance)"
            if any(v == "likelihood" for v in primary.values()) else "")
    if acc_models and any(m not in acc_models for m in r.model):
        note += "; [no accuracy] = likelihood scores only, no accuracy data"
    if r.coef_contaminated.isna().any():
        note += "\nnot estimable = contaminated stratum empty or too few rows (e.g. no Pile-contaminated item)"
    ctrl = r["controls"].astype(str) if "controls" in r else pd.Series([], dtype=str)
    if len(ctrl) and ctrl.str.contains("ref_").all():
        adj = "length and reference model"
    elif len(ctrl) and ctrl.str.contains("ref_").any():
        adj = "length (and reference model where available)"
    else:
        adj = "length"
    fig.suptitle(f"Contaminated minus not, adjusted for {adj} (95% CI)" + note, fontsize=8, y=1.02)
    fig.tight_layout()
    save(fig, out_dir, "fig_loglik_effect")


def order_strata(df):
    df = df.copy()
    df["_o"] = df.stratum.map({s: i for i, s in enumerate(STRATA_ORDER)})
    return df.sort_values(["benchmark", "model", "_o"]).drop(columns="_o")


def read(path, required):
    if not os.path.exists(path):
        print(f"  missing {path}: skipped")
        return None
    df = pd.read_csv(path)
    lack = set(required) - set(df.columns)
    if lack:
        sys.exit(f"error: {path} lacks columns {sorted(lack)}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default=".")
    ap.add_argument("--out-dir", default="paper_outputs")
    ap.add_argument("--decimals", type=int, default=3, help="decimals for accuracies/sensitivities")
    ap.add_argument("--chance-margin", type=float, default=0.05,
                    help="accuracy is 'near chance' if its CI upper bound <= chance + margin")
    ap.add_argument("--suffix", default="", help="read <name><suffix>.csv, e.g. _sensitivity")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    p = lambda f: os.path.join(a.in_dir, f.replace(".csv", f"{a.suffix}.csv"))  # noqa: E731
    cap = "Sensitivity models only. " if a.suffix else ""

    acc = read(p("strata_accuracy.csv"), ["model", "benchmark", "stratum", "n", "acc", "ci_lo", "ci_hi"])
    sens = read(p("sensitivity.csv"), ["model", "benchmark", "stratum", "sensitivity", "ci_lo", "ci_hi"])
    reg = read(p("regression.csv"), ["model", "benchmark", "n"])
    adj = read(p("adjusted_scores.csv"), ["benchmark", "model", "raw_acc", "adjusted_acc"])
    llreg = read(p("loglik_regression.csv"), ["model", "benchmark", "score", "n"])
    if all(x is None for x in (acc, sens, reg, adj, llreg)):
        sys.exit(f"error: no analyze.py outputs found in {os.path.abspath(a.in_dir)}")
    out = lambda f: os.path.join(a.out_dir, f)  # noqa: E731

    primary = primary_map(acc)
    scope = model_scope_note(acc, llreg)  # e.g. accuracy for pythia-410m only; others likelihood only
    if scope:
        print("NOTE: " + plain(scope))
    sc = scope + " " if scope else ""
    acc_models = sorted(acc.model.unique()) if acc is not None else []
    write_manifest("report.py", a, inputs=[p(f) for f in ("strata_accuracy.csv", "sensitivity.csv", "regression.csv",
                                                          "adjusted_scores.csv", "loglik_regression.csv")])
    if acc is not None:
        dodge_plot(acc, "acc", "ci_lo", "ci_hi", STRATA_ORDER, "accuracy", a.out_dir, "fig_accuracy_by_stratum",
                   primary=primary, note=scope)
        write_table(order_strata(acc), ["benchmark", "model", "stratum", "n", "acc", "ci_lo", "ci_hi"], HEADER,
                    cap + "Accuracy by overlap stratum (bootstrap 95\\% CI). Stratum `all' = evaluated items with an overlap "
                    "value for the model's corpus (see the coverage table). " + sc, f"tab:strata_accuracy{a.suffix}",
                    out("tab_strata_accuracy.tex"), a.decimals)
    if adj is not None:
        fig_raw_vs_adjusted(adj, acc, a.out_dir, note=scope)
        write_table(adj.sort_values(["benchmark", "model"]),
                    ["benchmark", "model", "raw_acc", "adjusted_acc", "inflation"], HEADER,
                    cap + "Raw (all items with an overlap value) vs contamination-adjusted (clean-subset) accuracy. " + sc, f"tab:adjusted{a.suffix}",
                    out("tab_adjusted_scores.tex"), a.decimals)
    if sens is not None:
        dodge_plot(sens, "sensitivity", "ci_lo", "ci_hi", STRATA_ORDER[:3],
                   "sensitivity (raw $-$ perturbed)", a.out_dir, "fig_sensitivity", zero_line=True, primary=primary,
                   note=scope)
        write_table(order_strata(sens), ["benchmark", "model", "stratum", "n", "raw", "perturbed", "sensitivity",
                                         "ci_lo", "ci_hi"], HEADER,
                    cap + "Contamination sensitivity: raw minus perturbed accuracy (bootstrap 95\\% CI). " + sc,
                    f"tab:sensitivity{a.suffix}", out("tab_sensitivity.tex"), a.decimals)
    if reg is not None:
        fig_forest(reg, a.out_dir, primary, note=scope)
        write_table(regression_display(reg).sort_values(["benchmark", "model"]),
                    ["benchmark", "model", "n", "odds_ratio_contaminated", "or_ci_lo", "or_ci_hi", "p", "p_holm",
                     "fisher_or", "fisher_ci_lo", "fisher_ci_hi", "fisher_p", "skipped_reason"], HEADER,
                    cap + "Logistic regression of correctness on contamination with controls (Wald 95\\% CI; "
                    "$p$ Holm over converged fits), and Fisher exact test of contaminated vs clean items "
                    "(conditional OR, exact 95\\% CI). Fits that did not converge are not estimable. "
                    + reg_controls_caption(reg) + sc,
                    f"tab:regression{a.suffix}", out("tab_regression.tex"), a.decimals)
    if llreg is not None:
        fig_loglik(llreg, a.out_dir, primary, acc_models)
        write_table(llreg.sort_values(["benchmark", "model", "score"]),
                    ["benchmark", "model", "score", "n", "coef_contaminated", "ci_lo", "ci_hi", "p", "p_holm", "p_bh"],
                    HEADER,
                    cap + "Likelihood memorisation signals: OLS coefficient of contamination (HC3 95\\% CI), "
                    + ("controlling for length and the reference model's score."
                       if "controls" in llreg and llreg.controls.astype(str).str.contains("ref_").all()
                       else "controlling for length (reference-model score only where shown in the controls)."
                       if "controls" in llreg and llreg.controls.astype(str).str.contains("ref_").any()
                       else "controlling for length.")
                    + " Rows without a coefficient could not be fitted (contaminated stratum empty or too few rows). "
                    + sc,
                    f"tab:loglik{a.suffix}", out("tab_loglik_regression.tex"), a.decimals)
    if acc is not None:
        sup = outcome_support(acc, llreg, a.chance_margin)
        sup.to_csv(out("outcome_support.csv"), index=False)
        write_table(sup, ["benchmark", "model", "acc", "ci_hi", "chance", "near_chance", "loglik_available",
                          "loglik_estimable", "primary_outcome"], HEADER,
                    cap + f"Outcome each model supports. Accuracy is near chance when its 95\\% CI upper bound is "
                    f"$\\le$ chance + {a.chance_margin:g}; contamination can then only be assessed through likelihood. "
                    "`n/a' = no accuracy data for this model. " + sc,
                    f"tab:outcomes{a.suffix}", out("tab_outcome_support.tex"), a.decimals)
        print("\n== Outcome each model supports ==")
        print(sup[["benchmark", "model", "acc", "ci_hi", "near_chance", "primary_outcome"]].round(3).to_string(index=False))
    cov_path = os.path.join(a.in_dir, "coverage.csv")
    if not a.suffix and os.path.exists(cov_path):
        cov = pd.read_csv(cov_path)
        for c in cov.columns:
            if c.startswith("items_overlap_computed_"):
                HEADER.setdefault(c, f"Overlap ({tex_escape(c.rsplit('_', 1)[1])})")
        write_table(cov, list(cov.columns), HEADER,
                    "Coverage: items per benchmark at each stage (sampled, with overlap computed per corpus, "
                    "evaluated, perturbable, scored for likelihood). Item counts are pooled over models; the "
                    "model columns give how many models each count comes from (likelihood: reference model "
                    "included). " + sc, "tab:coverage", out("tab_coverage.tex"), 0)
    av_path = os.path.join(a.in_dir, "data_availability.csv")
    if not a.suffix and os.path.exists(av_path):
        av = pd.read_csv(av_path)
        write_table(av, list(av.columns), HEADER,
                    "Data available per model, and whether the reference-model control is available for its "
                    "accuracy and likelihood regressions.", "tab:data_availability",
                    out("tab_data_availability.tex"), 0)
        print("\n== Data available per model ==")
        print(av.to_string(index=False))
    print(f"outputs in {os.path.abspath(a.out_dir)}")


if __name__ == "__main__":
    main()
