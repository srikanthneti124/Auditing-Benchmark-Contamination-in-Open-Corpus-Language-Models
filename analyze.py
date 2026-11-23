"""Step 3: stratified accuracy, contamination sensitivity, regression with
controls, contamination-adjusted score, ranking stability.

Strata (per corpus the model was trained on):
  clean        overlap == 0
  partial      0 < overlap < thr
  contaminated overlap >= thr

Optional --loglik itemloglik.csv: likelihood memorisation signals by stratum
(loglik_strata.csv), and OLS (HC3 robust SEs)
  score ~ contaminated + log_len [+ ref_score]
per model x benchmark x score (loglik_regression.csv), where ref_score is the
--ref-model's score on the same text.

Input validation (before any statistics; exits with an error message):
duplicate (model, benchmark, id, variant) rows in --results; different id sets
across models for the same benchmark and variant; a corpus needed by a model
that is missing from --overlap. Prints n per stratum per model x benchmark and
flags n < 30.

--model-corpus-map models.json: {model: {corpus, role, dtype}} (see
modelcfg.py; the older flat {model: corpus, "reference": model} form is still
read). Models not in the map fall back to the name-based rule (pythia -> pile,
olmo -> dolma).

Model names must carry a dtype suffix (EleutherAI/pythia-1.4b@fp32, written
by evaluate.py / itemloglik.py) that matches the dtype in models.json. Rows of
different dtypes for the same model are never analysed together: an analysis
set containing two dtypes of one model is an error. --ref-model accepts the
name with or without the suffix.

Models with role "sensitivity" (e.g. pythia-2.8b in fp16) are excluded by
default. --include-sensitivity analyses them in a separate pass (with the
reference model) and writes *_sensitivity.csv files; the main outputs are
unchanged.

Logistic regressions that cannot be fitted (one stratum only, or a constant
outcome) are no longer dropped silently: a warning is printed and
regression.csv gets a row with skipped_reason. Corpus x benchmark pairs with
no contaminated item are listed as clean-only negative controls.

Overlap rows with complete=False (overlap.py --allow-partial) are excluded by
default (--include-incomplete keeps them); the excluded pairs are printed.

Additive outputs (no existing column changes):
  strata_accuracy.csv  appended column primary_outcome (accuracy | likelihood)
                       per model x benchmark: "likelihood" when the upper 95%
                       CI bound of overall accuracy is <= chance + --chance-margin
                       (the near-chance rule of report.py), else "accuracy".
  loglik_regression.csv  appended columns p_holm and p_bh: Holm (family-wise
                       error) and Benjamini-Hochberg (false discovery rate)
                       adjusted p. Correction family = all tests in that file
                       (every model x benchmark x score with a p-value); the
                       *_sensitivity file is its own family. Raw p unchanged.
  regression.csv       appended: converged (optimiser converged, no perfect
                       separation of `contaminated`, finite standard errors,
                       |log OR| <= 15), n_contaminated, n_clean, Fisher exact
                       test contaminated vs clean (fisher_or = conditional MLE,
                       exact 95% CI, two-sided fisher_p; partial items not in
                       the 2x2), p_holm / p_bh over the converged fits only.
                       Rows that did not converge have EMPTY odds ratio, CI and
                       p, and skipped_reason says why.
  coverage.csv         per benchmark: items sampled (rows in the overlap file),
                       items with overlap computed per corpus (non-missing and
                       complete), items evaluated (raw rows), items perturbable
                       (perturbed rows), items with likelihood scores; appended
                       models_accuracy / models_loglik = number of models the
                       pooled item counts come from (models_loglik includes the
                       reference model).
  data_availability.csv  per model: role, corpus, whether it has accuracy,
                       raw - perturbed sensitivity and likelihood rows, and
                       whether the reference-model control is available for
                       its accuracy and likelihood regressions.
  regression.csv       appended column controls. If --ref-model has no rows in
                       --results (likelihood-only reference), the accuracy
                       regression is fitted without ref_correct, as when no
                       --ref-model is given, and controls says so.

Models present only in --loglik (no results.csv rows) get likelihood outputs
only. --loglik is validated like --results: duplicate (model, benchmark, id)
rows, id sets differing across models (reference included), missing corpora.
Stratum "all" = evaluated items that have an overlap value for the model's
corpus (MMLU items without any n-gram window are in no stratum).

Usage:
  python analyze.py --results results.csv --overlap overlap.csv --ref-model Qwen/Qwen2.5-1.5B
  python analyze.py --results results.csv --overlap overlap.csv --ref-model Qwen/Qwen2.5-1.5B --loglik itemloglik.csv
  python analyze.py --model-corpus-map models.json --ref-model Qwen/Qwen2.5-1.5B --loglik itemloglik.csv
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact, kendalltau, spearmanr
from scipy.stats.contingency import odds_ratio
from statsmodels.stats.multitest import multipletests

from manifest import write_manifest
from modelcfg import DTYPE_TAGS, base_model, dtype_tag, load_models, lookup, strip_dtype

LL_SCORES = ["ll_question", "ll_question_answer", "mink_question", "mink_question_answer"]
KEY = ["model", "benchmark", "id", "variant"]
MIN_N = 30
CHANCE = {"mmlu": 0.25, "gsm8k": 0.0, "humaneval": 0.0}  # chance accuracy per benchmark (also used by report.py)
MODEL_CORPUS = {}  # filled from --model-corpus-map
MODEL_CFG = {}  # filled from --model-corpus-map: {model: {corpus, role, dtype}}


def load_corpus_map(path):
    return {k: c["corpus"] for k, c in load_models(path).items() if c["role"] != "reference"}


def role_of(model):
    c = lookup(model, MODEL_CFG)
    return c["role"] if c else "primary"


def corpus_of(model):
    for key in (model, strip_dtype(model), base_model(model)):
        if key in MODEL_CORPUS:
            return MODEL_CORPUS[key]
    m = model.lower()
    if "pythia" in m:
        return "pile"
    if "olmo" in m:
        return "dolma"
    return None


def assign_stratum(overlap, thr):
    overlap = np.asarray(overlap, float)
    return np.where(overlap == 0, "clean", np.where(overlap >= thr, "contaminated", "partial"))


def validate_inputs(res, ov, ref_model=None):
    """Raise SystemExit on inconsistent inputs. Statistics are not touched."""
    errors = []
    dup = res[res.duplicated(KEY, keep=False)]
    if len(dup):
        errors.append(f"{len(dup)} rows in results share a (model, benchmark, id, variant) key, e.g.\n"
                      + dup.sort_values(KEY).head(6).to_string(index=False))
    for (b, v), g in res.groupby(["benchmark", "variant"]):
        sets = {m: set(gm.id) for m, gm in g.groupby("model")}
        if len({frozenset(s) for s in sets.values()}) > 1:
            union = set().union(*sets.values())
            detail = ", ".join(f"{m}: {len(s)} ids, missing {sorted(union - s)[:5]}" for m, s in sets.items()
                               if s != union)
            errors.append(f"id sets differ across models for {b}/{v} ({len(union)} ids in union): {detail}")
    models = [m for m in res.model.unique() if m != ref_model]
    needed = {corpus_of(m) for m in models} - {None}
    missing = sorted(needed - set(ov.corpus.unique()))
    if missing:
        users = [m for m in models if corpus_of(m) in missing]
        errors.append(f"overlap file lacks corpus {missing} needed by {users}")
    if errors:
        raise SystemExit("error: invalid inputs\n- " + "\n- ".join(errors))
    unmapped = [m for m in models if corpus_of(m) is None]
    if unmapped:
        print(f"warning: no corpus for {unmapped}; excluded (add them to --model-corpus-map)")


def validate_loglik(ll, ov, ref_model=None, path="itemloglik.csv"):
    """Same checks as validate_inputs, for the likelihood file: duplicate (model, benchmark, id) rows; id sets
    that differ across models (the reference model included, since every likelihood regression is merged with
    it); a corpus needed by a model that is missing from the overlap file. Raise SystemExit; statistics untouched."""
    errors = []
    k = ["model", "benchmark", "id"]
    dup = ll[ll.duplicated(k, keep=False)]
    if len(dup):
        errors.append(f"{len(dup)} rows in {path} share a (model, benchmark, id) key, e.g.\n"
                      + dup.sort_values(k).head(6)[k].to_string(index=False))
    for b, g in ll.groupby("benchmark"):
        sets = {m: set(gm.id) for m, gm in g.groupby("model")}
        if len({frozenset(s) for s in sets.values()}) > 1:
            union = set().union(*sets.values())
            detail = ", ".join(f"{m}: {len(s)} ids, missing {sorted(union - s)[:5]}" for m, s in sets.items()
                               if s != union)
            errors.append(f"{path}: id sets differ across models for {b} ({len(union)} ids in union): {detail}")
    models = [m for m in ll.model.unique() if m != ref_model]
    needed = {corpus_of(m) for m in models} - {None}
    missing = sorted(needed - set(ov.corpus.unique()))
    if missing:
        errors.append(f"overlap file lacks corpus {missing} needed by "
                      f"{[m for m in models if corpus_of(m) in missing]} in {path}")
    if errors:
        raise SystemExit("error: invalid inputs\n- " + "\n- ".join(errors))


def data_availability(res, ll, ref_model, ll_ref):
    """Which data each model has. Additive output data_availability.csv; nothing is computed from it."""
    rows = []
    for m in sorted(set(res.model) | (set(ll.model) if ll is not None else set())):
        r = res[res.model == m]
        is_ref = m in (ref_model, ll_ref)
        rows.append({
            "model": m, "role": "reference" if is_ref else role_of(m), "corpus": corpus_of(m) if not is_ref else "",
            "accuracy": "yes" if (r.variant == "raw").any() else "no",
            "sensitivity_raw_minus_perturbed": "yes" if (r.variant == "perturbed").any() else "no",
            "likelihood": "yes" if ll is not None and (ll.model == m).any() else "no",
        })
    av = pd.DataFrame(rows)
    has_ref_acc = ref_model is not None and (res.model == ref_model).any()
    has_ref_ll = ll is not None and ll_ref is not None and (ll.model == ll_ref).any()
    av["reference_control_accuracy"] = [
        "n/a (reference)" if r == "reference" else
        ("yes" if has_ref_acc else "no: reference model has no results.csv rows") if a == "yes" else "n/a (no accuracy)"
        for r, a in zip(av.role, av.accuracy)]
    av["reference_control_likelihood"] = [
        "n/a (reference)" if r == "reference" else
        ("yes" if has_ref_ll else "no: reference model has no likelihood rows") if l == "yes" else "n/a (no likelihood)"
        for r, l in zip(av.role, av.likelihood)]
    return av


def add_primary_outcome(acc, margin=0.05):
    """Appends primary_outcome per model x benchmark: 'likelihood' if the 'all' row's ci_hi <= chance + margin."""
    a = acc[acc.stratum == "all"]
    near = {(m, b): bool(hi <= CHANCE.get(b, np.nan) + margin) for m, b, hi in zip(a.model, a.benchmark, a.ci_hi)}
    acc = acc.copy()
    acc["primary_outcome"] = ["likelihood" if near.get((m, b)) else "accuracy" for m, b in zip(acc.model, acc.benchmark)]
    return acc


def add_multiplicity(rg):
    """Appends p_holm and p_bh; the family is every row of rg with a p-value."""
    rg = rg.copy()
    rg["p_holm"] = np.nan
    rg["p_bh"] = np.nan
    if "p" in rg and rg.p.notna().any():
        m = rg.p.notna()
        rg.loc[m, "p_holm"] = multipletests(rg.p[m], method="holm")[1]
        rg.loc[m, "p_bh"] = multipletests(rg.p[m], method="fdr_bh")[1]
    return rg


def coverage(res, ov_all, ov, ll=None):
    """Items per benchmark at each stage, for the Methods section."""
    rows = []
    for b in sorted(set(ov_all.benchmark) | set(res.benchmark)):
        o_all, o, r = ov_all[ov_all.benchmark == b], ov[ov.benchmark == b], res[res.benchmark == b]
        row = {"benchmark": b, "items_sampled": int(o_all.id.nunique())}
        for c in sorted(ov_all.corpus.unique()):
            row[f"items_overlap_computed_{c}"] = int(o[(o.corpus == c) & o.overlap.notna()].id.nunique())
        row["items_evaluated"] = int(r[r.variant == "raw"].id.nunique())
        row["items_perturbable"] = int(r[r.variant == "perturbed"].id.nunique())
        if ll is not None:
            row["items_loglik"] = int(ll[ll.benchmark == b].id.nunique())
        # appended: how many models the pooled item counts above come from
        row["models_accuracy"] = int(r[r.variant == "raw"].model.nunique())
        if ll is not None:
            row["models_loglik"] = int(ll[ll.benchmark == b].model.nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def exclude_incomplete(ov, include=False):
    """overlap.py --allow-partial marks rows of incomplete benchmark x corpus pairs complete=False;
    they are dropped unless include. No "complete" column means complete."""
    if "complete" not in ov.columns:
        return ov
    ok = ov.complete.map(lambda v: str(v).strip().lower() in ("true", "1", "1.0"))
    bad = ov[~ok].groupby(["benchmark", "corpus"]).size()
    if len(bad):
        print(f"note: {len(bad)} benchmark x corpus pair(s) incomplete in the overlap file "
              f"({'kept: --include-incomplete' if include else 'EXCLUDED'}): "
              + ", ".join(f"{b}/{c} ({n} items; overlap.py had windows remaining, see overlap_partial_notice.txt)"
                          for (b, c), n in bad.items()))
    return ov if include else ov[ok]


def check_model_names(df, path):
    """Every model name carries a dtype suffix that matches models.json."""
    names = sorted(df.model.unique())
    bare = [m for m in names if dtype_tag(m) is None]
    if bare:
        raise SystemExit(f"error: {path}: model names without a dtype suffix (@fp32/@fp16/@bf16): {bare[:5]}; "
                         f"re-run evaluate.py / itemloglik.py")
    for m in names:
        c = lookup(m, MODEL_CFG)
        if c and DTYPE_TAGS[c["dtype"]] != dtype_tag(m):
            raise SystemExit(f"error: {path}: {m} was computed in {dtype_tag(m)}, but models.json lists "
                             f"{strip_dtype(m)} as {c['dtype']}")


def resolve_model(names, query):
    """Exact model name for --ref-model given with or without its dtype suffix."""
    if query is None:
        return None
    names = set(names)
    if query in names:
        return query
    hits = sorted(n for n in names if strip_dtype(n) == query)
    if len(hits) > 1:
        raise SystemExit(f"error: --ref-model {query} matches several dtypes {hits}; pass one of them exactly")
    return hits[0] if hits else query


def check_no_pooling(df):
    """Rows of different dtypes for the same model must never be analysed together."""
    names = pd.Series(df.model.unique())
    multi = names.groupby(names.map(strip_dtype)).apply(sorted)
    multi = multi[multi.map(len) > 1]
    if len(multi):
        raise SystemExit("error: rows of different dtypes for the same model in one analysis: "
                         + "; ".join(", ".join(v) for v in multi)
                         + ". Give all but one role \"sensitivity\" in models.json.")


def select_role(df, sensitivity, ref_model):
    """Rows of sensitivity models (sensitivity=True) or of all other models; the reference is always kept."""
    is_sens = df.model.map(role_of) == "sensitivity"
    return df[(is_sens == sensitivity) | (df.model == ref_model)].copy()


def regression_skip_reason(d):
    if d.empty:
        return "no rows (none left after merging with the reference model)"
    if d.contaminated.nunique() < 2:
        return ("one stratum only: no contaminated items (overlap >= thr)" if d.contaminated.iloc[0] == 0
                else "one stratum only: every item contaminated")
    if d.correct.nunique() < 2:
        return f"constant outcome: every item {'correct' if d.correct.iloc[0] else 'incorrect'}"
    return None


REG_DIAG_COLS = ["converged", "n_contaminated", "n_clean", "fisher_or", "fisher_ci_lo", "fisher_ci_hi", "fisher_p",
                 "p_holm", "p_bh"]
MAX_ABS_COEF = 15.0  # |log OR| beyond this is treated as divergence (OR outside about [3e-7, 3e6])


def separation_reason(d):
    """Quasi-complete separation of `contaminated`: the outcome is constant within one group."""
    for flag, name in ((1, "contaminated"), (0, "non-contaminated")):
        y = d.correct[d.contaminated == flag]
        if len(y) and y.nunique() == 1:
            return (f"perfect separation: all {len(y)} {name} items "
                    f"{'correct' if y.iloc[0] else 'incorrect'}")
    return None


def fit_diagnostics(fit, d, caught=()):
    """(converged, reason). False when the optimiser did not converge, the data separate, or the
    contaminated coefficient has an infinite / missing standard error or diverges."""
    sep = separation_reason(d)
    if sep:
        return False, sep
    if not bool(getattr(fit, "mle_retvals", {}).get("converged", True)):
        return False, "maximum likelihood did not converge"
    names = {w.category.__name__ for w in caught}
    if "PerfectSeparationWarning" in names or "PerfectSeparationError" in names:
        return False, "perfect separation reported by statsmodels"
    se = fit.bse.get("contaminated", np.nan)
    if not np.isfinite(se) or not np.all(np.isfinite(fit.bse)):
        return False, "infinite or missing standard errors"
    if abs(fit.params["contaminated"]) > MAX_ABS_COEF:
        return False, "coefficient diverged (|log OR| > 15)"
    return True, ""


def fisher_columns(d):
    """Fisher exact test of correctness, contaminated (overlap >= thr) vs clean (overlap == 0) items;
    partial items are not in this 2x2 table. OR = conditional MLE with exact 95% CI; p two-sided.
    Empty when either group has no items."""
    con, cln = d[d.contaminated == 1], d[d.overlap == 0]
    out = {"n_contaminated": int(len(con)), "n_clean": int(len(cln)),
           "fisher_or": np.nan, "fisher_ci_lo": np.nan, "fisher_ci_hi": np.nan, "fisher_p": np.nan}
    if len(con) and len(cln):
        table = [[int(con.correct.sum()), int(len(con) - con.correct.sum())],
                 [int(cln.correct.sum()), int(len(cln) - cln.correct.sum())]]
        res = odds_ratio(table, kind="conditional")
        ci = res.confidence_interval(confidence_level=0.95)
        out.update(fisher_or=float(res.statistic), fisher_ci_lo=float(ci.low), fisher_ci_hi=float(ci.high),
                   fisher_p=float(fisher_exact(table, alternative="two-sided")[1]))
    return out


def finish_regression(reg, extra):
    """Blank OR / CI / p of non-converged fits (with skipped_reason saying why), then append the
    diagnostic, Fisher and multiplicity columns. Existing columns of converged rows are untouched."""
    ex = pd.DataFrame(extra, index=reg.index)
    bad = ~ex.converged.astype(bool)
    why = ex["_why"]
    for c in ("odds_ratio_contaminated", "or_ci_lo", "or_ci_hi", "p"):
        if c in reg:
            reg.loc[bad, c] = np.nan
    fill = bad & why.notna() & (why.astype(str).str.strip() != "")
    reg["skipped_reason"] = reg.skipped_reason.astype(object)
    reg.loc[fill, "skipped_reason"] = "not estimable: " + why[fill]
    for c in ("converged", "n_contaminated", "n_clean", "fisher_or", "fisher_ci_lo", "fisher_ci_hi", "fisher_p"):
        reg[c] = ex[c].to_numpy()
    reg["converged"] = reg.converged.astype(bool)
    reg["p_holm"] = np.nan
    reg["p_bh"] = np.nan
    if "p" in reg:
        fam = reg.converged & reg.p.notna()
        if fam.any():
            reg.loc[fam, "p_holm"] = multipletests(reg.p[fam], method="holm")[1]
            reg.loc[fam, "p_bh"] = multipletests(reg.p[fam], method="fdr_bh")[1]
    return reg


def negative_controls(raw):
    """Corpus x benchmark pairs with no contaminated item (clean-only)."""
    rows = []
    for (c, b), g in raw.groupby(["corpus", "benchmark"]):
        n = g.drop_duplicates("id").stratum.value_counts()
        if n.get("contaminated", 0) == 0:
            rows.append({"corpus": c, "benchmark": b, "clean": int(n.get("clean", 0)),
                         "partial": int(n.get("partial", 0)), "models": sorted(g.model.unique())})
    return rows


def stratum_counts(raw):
    """n per stratum per model x benchmark (raw variant), with an n < MIN_N flag."""
    n = raw.groupby(["model", "benchmark", "stratum"]).size().unstack("stratum", fill_value=0)
    for s in ("clean", "partial", "contaminated"):
        if s not in n:
            n[s] = 0
    n = n[["clean", "partial", "contaminated"]]
    n["flag"] = [", ".join(f"{s} n={n.loc[i, s]}<{MIN_N}" for s in ("clean", "partial", "contaminated")
                           if n.loc[i, s] < MIN_N) for i in n.index]
    return n.reset_index()


def boot_ci(x, B=1000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    idx = np.random.default_rng(seed).integers(0, len(x), (B, len(x)))
    return tuple(np.percentile(x[idx].mean(1), [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--overlap", default="overlap.csv")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--ref-model", default=None, help="model not trained on the corpora; used as difficulty control")
    ap.add_argument("--loglik", default=None, help="itemloglik.csv; enables the likelihood analysis")
    ap.add_argument("--model-corpus-map", default=None, help="models.json {model: {corpus, role, dtype}}; see docstring")
    ap.add_argument("--include-sensitivity", action="store_true",
                    help="also analyse role=sensitivity models, separately, into *_sensitivity.csv")
    ap.add_argument("--include-incomplete", action="store_true",
                    help="keep overlap rows with complete=False (overlap.py --allow-partial); default: excluded")
    ap.add_argument("--chance-margin", type=float, default=0.05,
                    help="primary_outcome is 'likelihood' if the accuracy CI upper bound <= chance + margin")
    a = ap.parse_args()

    if a.model_corpus_map:
        MODEL_CFG.update(load_models(a.model_corpus_map))
        MODEL_CORPUS.update(load_corpus_map(a.model_corpus_map))
    res = pd.read_csv(a.results)
    ov_all = pd.read_csv(a.overlap)
    check_model_names(res, a.results)
    ref_model = resolve_model(res.model, a.ref_model)
    validate_inputs(res, ov_all, ref_model)
    ov = exclude_incomplete(ov_all, a.include_incomplete)
    ll = None
    if a.loglik:
        ll = pd.read_csv(a.loglik)
        missing = {"model", "benchmark", "id"} - set(ll.columns)
        if missing:
            raise SystemExit(f"error: {a.loglik} lacks columns {sorted(missing)}")
        check_model_names(ll, a.loglik)
        validate_loglik(ll, ov_all, resolve_model(ll.model, a.ref_model), a.loglik)
    write_manifest("analyze.py", a, inputs=[a.results, a.overlap, a.loglik, a.model_corpus_map])
    ll_ref_name = resolve_model(ll.model, a.ref_model) if ll is not None else None
    av = data_availability(res, ll, ref_model, ll_ref_name)
    av.to_csv("data_availability.csv", index=False)
    print("\n== Data available per model (data_availability.csv) ==")
    print(av.to_string(index=False))
    acc_models = av[(av.accuracy == "yes") & (av.role != "reference")].model.tolist()
    ll_only = av[(av.accuracy == "no") & (av.likelihood == "yes") & (av.role != "reference")].model.tolist()
    if ll_only:
        print(f"NOTE: accuracy, sensitivity (raw - perturbed) and the accuracy regression exist only for "
              f"{acc_models}; {ll_only} have likelihood scores only.")
    if ref_model and not (res.model == ref_model).any():
        print(f"NOTE: reference model {a.ref_model} has no rows in {a.results}: the accuracy regression is fitted "
              f"WITHOUT the ref_correct control (column 'controls' in regression.csv). The likelihood regressions "
              f"use the reference model's likelihood scores where it has them (column 'controls').")
    cov = coverage(res, ov_all, ov, ll)
    cov.to_csv("coverage.csv", index=False)
    print("\n== Coverage: items per benchmark at each stage (coverage.csv) ==")
    print(cov.to_string(index=False))

    sens_models = sorted(m for m in res.model.unique() if role_of(m) == "sensitivity")
    passes = [("", False)]
    if sens_models and a.include_sensitivity:
        passes.append(("_sensitivity", True))
    elif sens_models:
        print(f"note: sensitivity models {sens_models} excluded (--include-sensitivity analyses them "
              f"separately into *_sensitivity.csv)")
    for suffix, sens in passes:
        if sens:
            print(f"\n######## sensitivity models {sens_models}: separate outputs *{suffix}.csv ########")
        r = select_role(res, sens, ref_model)
        check_no_pooling(r)
        analyze_results(r, ov, ref_model, a.thr, suffix, a.chance_margin)
        if ll is not None:
            ll_ref = resolve_model(ll.model, a.ref_model)
            l = select_role(ll, sens, ll_ref)
            check_no_pooling(l)
            loglik_analysis(l, ov, ll_ref, a.thr, a.loglik, suffix)


def analyze_results(res, ov, ref_model, thr, suffix="", chance_margin=0.05):
    out = lambda name: f"{name}{suffix}.csv"  # noqa: E731
    ref = None
    if ref_model:
        ref = (
            res[(res.model == ref_model) & (res.variant == "raw")][["benchmark", "id", "correct"]]
            .rename(columns={"correct": "ref_correct"})
        )
        if ref.empty:  # reference model has likelihood rows only: same model as without --ref-model, labelled
            print(f"\nwarning: --ref-model {ref_model} has no rows in results; accuracy regressions run WITHOUT "
                  f"ref_correct (see column 'controls')")
            ref = None
        res = res[res.model != ref_model]

    res["corpus"] = res.model.map(corpus_of)
    res = res.dropna(subset=["corpus"])
    df = res.merge(ov, on=["corpus", "benchmark", "id"]).dropna(subset=["overlap"])
    df["stratum"] = assign_stratum(df.overlap, thr)
    raw = df[df.variant == "raw"].copy()

    counts = stratum_counts(raw)
    counts.to_csv(out("strata_n"), index=False)
    print(f"\n== n per stratum (raw variant); flag = n < {MIN_N} ==")
    print(counts.to_string(index=False))

    # 1. accuracy by stratum
    rows = []
    for (m, b), g in raw.groupby(["model", "benchmark"]):
        for s, gs in list(g.groupby("stratum")) + [("all", g)]:
            lo, hi = boot_ci(gs.correct)
            rows.append({"model": m, "benchmark": b, "stratum": s, "n": len(gs),
                         "acc": gs.correct.mean(), "ci_lo": lo, "ci_hi": hi})
    acc = add_primary_outcome(pd.DataFrame(rows), chance_margin) if rows else pd.DataFrame(rows)
    acc.to_csv(out("strata_accuracy"), index=False)
    print("\n== Accuracy by overlap stratum ==")
    print(acc.round(3).to_string(index=False))
    print("note: stratum 'all' = evaluated items that have an overlap value for the model's corpus "
          "(items without any n-gram window are not in any stratum; compare n with coverage.csv)")

    # 2. contamination sensitivity = raw - perturbed accuracy, per stratum
    if (df.variant == "perturbed").any():
        wide = df.pivot_table(index=["model", "benchmark", "id", "stratum"], columns="variant", values="correct").dropna()
        wide["drop"] = wide["raw"] - wide["perturbed"]
        rows = []
        for (m, b, s), g in wide.groupby(["model", "benchmark", "stratum"]):
            lo, hi = boot_ci(g["drop"])
            rows.append({"model": m, "benchmark": b, "stratum": s, "n": len(g),
                         "raw": g["raw"].mean(), "perturbed": g["perturbed"].mean(),
                         "sensitivity": g["drop"].mean(), "ci_lo": lo, "ci_hi": hi})
        sens = pd.DataFrame(rows)
        sens.to_csv(out("sensitivity"), index=False)
        print("\n== Contamination sensitivity (raw - perturbed) ==")
        print(sens.round(3).to_string(index=False))

    # 3. logistic regression with controls
    rows, extra = [], []  # extra: appended diagnostic columns, kept apart so the existing column order is unchanged
    for (m, b), g in raw.groupby(["model", "benchmark"]):
        d = g.copy()
        d["contaminated"] = (d.overlap >= thr).astype(int)
        d["log_len"] = np.log(d.n_words.clip(lower=1))
        formula = "correct ~ contaminated + log_len"
        if ref is not None:
            d = d.merge(ref, on=["benchmark", "id"])
            formula += " + ref_correct"
        fisher = fisher_columns(d)
        reason = regression_skip_reason(d)
        if reason:
            print(f"warning: regression skipped for {m} / {b}: {reason}")
            rows.append({"model": m, "benchmark": b, "n": len(d), "skipped_reason": reason})
            extra.append({"converged": False, "_why": None, **fisher})
            continue
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = smf.logit(formula, data=d).fit(disp=0)
            ci = np.exp(fit.conf_int().loc["contaminated"])  # Wald 95%
            rows.append({"model": m, "benchmark": b, "n": len(d),
                         "odds_ratio_contaminated": float(np.exp(fit.params["contaminated"])),
                         "or_ci_lo": float(ci.iloc[0]), "or_ci_hi": float(ci.iloc[1]),
                         "p": float(fit.pvalues["contaminated"])})
            ok, why = fit_diagnostics(fit, d, caught)
            if not ok:
                print(f"warning: regression for {m} / {b} not estimable: {why}")
            extra.append({"converged": ok, "_why": why, **fisher})
        except Exception as e:
            rows.append({"model": m, "benchmark": b, "n": len(d), "error": str(e)})
            extra.append({"converged": False, "_why": f"fit error: {e}", **fisher})
    reg = pd.DataFrame(rows)
    if "skipped_reason" not in reg:
        reg["skipped_reason"] = ""
    reg = finish_regression(reg, extra)
    reg["controls"] = "contaminated + log_len" + (" + ref_correct" if ref is not None else "")  # appended
    n_family = int((reg.converged & reg.p.notna()).sum()) if "p" in reg else 0
    print(f"\nregression.csv multiple comparisons: family = the {n_family} converged logistic fits in "
          f"regression{suffix}.csv; p_holm = Holm, p_bh = Benjamini-Hochberg; non-converged rows excluded (empty)")
    reg.to_csv(out("regression"), index=False)
    print("\n== Logistic regression: correct ~ contaminated + controls ==")
    print(reg.round(4).to_string(index=False))

    print(f"\n== Clean-only pairs: negative controls (no item has overlap >= {thr}; "
          f"no contamination effect is estimable) ==")
    controls = negative_controls(raw)
    for c in controls:
        print(f"{c['benchmark']} x {c['corpus']}: clean n={c['clean']}, partial n={c['partial']}, contaminated n=0; "
              f"report as a clean-only negative control for {', '.join(c['models'])}")
    if not controls:
        print("none")

    # 4. adjusted score and ranking stability
    piv = acc[acc.stratum.isin(["all", "clean"])].pivot_table(index=["benchmark", "model"], columns="stratum", values="acc")
    if "clean" in piv:
        piv = piv.rename(columns={"all": "raw_acc", "clean": "adjusted_acc"})
        piv["inflation"] = piv.raw_acc - piv.adjusted_acc
        piv.to_csv(out("adjusted_scores"))
        print("\n== Raw vs contamination-adjusted (clean-subset) accuracy ==")
        print(piv.round(3).to_string())
        for b, g in piv.dropna().groupby(level=0):
            if len(g) >= 3:
                tau, p = kendalltau(g.raw_acc, g.adjusted_acc)
                print(f"Kendall tau raw vs adjusted [{b}]: {tau:.3f} (p={p:.3f}, models={len(g)})")



def loglik_analysis(ll, ov, ref_model, thr, path, suffix=""):
    out = lambda name: f"{name}{suffix}.csv"  # noqa: E731
    scores = [s for s in LL_SCORES if s in ll.columns]
    ref = None
    if ref_model and (ll.model == ref_model).any():
        ref = ll[ll.model == ref_model][["benchmark", "id"] + scores].rename(
            columns={s: f"ref_{s}" for s in scores})
    elif ref_model:
        print(f"\nwarning: --ref-model {ref_model} not in {path}; regressions run without ref_score")
    ll = ll[ll.model != ref_model].copy()
    ll["corpus"] = ll.model.map(corpus_of)
    d = ll.dropna(subset=["corpus"]).merge(ov, on=["corpus", "benchmark", "id"]).dropna(subset=["overlap"])
    if d.empty:
        print("\nwarning: no loglik rows matched overlap.csv (check model names / corpora)")
        return
    d["stratum"] = assign_stratum(d.overlap, thr)
    d["contaminated"] = (d.overlap >= thr).astype(int)
    d["log_len"] = np.log(d.n_words.clip(lower=1))
    if ref is not None:
        d = d.merge(ref, on=["benchmark", "id"], how="left")

    strata, regs = [], []
    for (m, b), g in d.groupby(["model", "benchmark"]):
        for s in scores:
            rho, rho_p = spearmanr(g[s], g.overlap, nan_policy="omit") if g.overlap.nunique() > 1 else (np.nan, np.nan)
            for st, gs in list(g.groupby("stratum")) + [("all", g)]:
                lo, hi = boot_ci(gs[s].dropna())
                strata.append({"model": m, "benchmark": b, "score": s, "stratum": st, "n": int(gs[s].notna().sum()),
                               "mean": gs[s].mean(), "ci_lo": lo, "ci_hi": hi,
                               "spearman_rho_vs_overlap": rho, "spearman_p": rho_p})
            formula = f"{s} ~ contaminated + log_len"
            cols = [s, "contaminated", "log_len"]
            if ref is not None:
                formula += f" + ref_{s}"
                cols.append(f"ref_{s}")
            gg = g[cols].dropna()
            row = {"model": m, "benchmark": b, "score": s, "n": len(gg), "controls": formula.split("~")[1].strip()}
            if gg.contaminated.nunique() < 2 or len(gg) <= len(cols) + 1:
                row["error"] = "contaminated stratum empty or too few rows"
            else:
                try:
                    fit = smf.ols(formula, data=gg).fit(cov_type="HC3")
                    lo, hi = fit.conf_int().loc["contaminated"]
                    row.update(coef_contaminated=float(fit.params["contaminated"]), ci_lo=float(lo), ci_hi=float(hi),
                               p=float(fit.pvalues["contaminated"]),
                               higher_in_contaminated_p05=bool(fit.params["contaminated"] > 0 and fit.pvalues["contaminated"] < 0.05))
                except Exception as e:
                    row["error"] = str(e)
            regs.append(row)
    st = pd.DataFrame(strata)
    st.to_csv(out("loglik_strata"), index=False)
    print("\n== Likelihood signals by overlap stratum (mean, bootstrap 95% CI) ==")
    print(st.round(3).to_string(index=False))
    rg = add_multiplicity(pd.DataFrame(regs))
    print(f"\nmultiple comparisons: family = all {int(rg.p.notna().sum()) if 'p' in rg else 0} tests in "
          f"loglik_regression{suffix}.csv (model x benchmark x score); p_holm = Holm, p_bh = Benjamini-Hochberg")
    rg.to_csv(out("loglik_regression"), index=False)
    print("\n== OLS (HC3): score ~ contaminated + controls; coef = difference in mean log-prob ==")
    print(rg.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
