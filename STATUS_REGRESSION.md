# STATUS_REGRESSION (2026-10-29 04:31)

## Steps
| Step | Result | Evidence (this session) |
|---|---|---|
| 1 analyze.py fix | **PASS** | regression.csv gains appended columns `converged`, `n_contaminated`, `n_clean`, `fisher_or`, `fisher_ci_lo`, `fisher_ci_hi`, `fisher_p`, `p_holm`, `p_bh`. Non-converged rows have **empty** OR, CI and p, and `skipped_reason` says why. Holm/BH family = converged fits only; analyze prints "family = the 1 converged logistic fits in regression.csv". |
| 2 report.py and LaTeX | **PASS** | The forest plot draws converged fits only; others are labelled "not estimable". The table prints "not estimable" with the Fisher OR, CI and p beside it. Structural check: 7/7 .tex pass (`texcheck.py`; pdflatex compile UNVERIFIED). |
| 3 Tests | **PASS** | `tests/test_regression_fix.py`, 10 passed: perfect separation (all contaminated wrong), a normal converged fit equal to a plain `smf.logit` fit, empty CSV cells (not 0.0 / inf / 1.0), Fisher known values, Holm/BH family, report table and forest, old vs new. |
| 4 Verify | **PASS** | `python -m pytest tests -q`: **631 passed**, 0 failed (4 RuntimeWarnings: `overflow encountered in exp` while exponentiating the separated fit's CI, before that row is blanked). Smoke rerun `_work/smoke/limit20` (exit 0): the pythia-410m@fp32 MMLU row reads "not estimable … perfect separation: all 5 contaminated items incorrect", Fisher OR 0.00 [0.00, 54.60], p 1.000. Forest plot opened: only pythia-70m drawn. Real overlap.csv, results.csv and itemloglik.csv do not exist. |

## What changed
- **Convergence rule** (`analyze.fit_diagnostics`): not converged when any of these holds:
  - perfect separation (the outcome is constant within the contaminated or the non-contaminated items);
  - `mle_retvals["converged"]` is False;
  - statsmodels reports perfect separation;
  - any standard error is infinite or missing;
  - |log OR| > 15.
- **Fisher test** (`analyze.fisher_columns`): contaminated (overlap ≥ 0.5) vs **clean** (overlap = 0) items; partial items are not in the 2×2. Conditional MLE OR, exact 95% CI, two-sided p (scipy).
- **Old vs new** (frozen pre-fix copy `tests/legacy/analyze_before_regression_fix.py`):
  - On data without separation, every pre-existing column of every output file is identical.
  - On data with separation, only `regression.csv` differs, and only in the non-converged row: OR, CI and p are now empty, and `skipped_reason` was filled. That is the change you requested to those existing columns; converged rows are identical.
- **Files:**
  - Changed: `analyze.py`, `report.py`, `METHODS_DRAFT.md` (new regression bullets).
  - New: `tests/test_regression_fix.py`, `tests/legacy/analyze_before_regression_fix.py`.
  - Backups `*.bak_20261029_0423` / `_0428`; diffs in `_work/diffs_regfix/`.

## Cache
SHA-256 `17CB80FDBA17E0AFB4F583501893EC3A072059EA7A508B2EB71A4A69034B47C4`, mtime 2026-10-25 03:39:42: **identical at start (04:23) and end (04:31)**, matching the expected values.

## Caveats / UNVERIFIED
- **Fisher OR can't be filled everywhere.** It is undefined for a table with no correct (or no incorrect) items. The constant-outcome rows (GSM8K / HumanEval smoke) have an empty `fisher_or` with p 1.000 and CI [0, ∞]. With no contaminated or no clean items, all Fisher columns are empty.
- **Fisher reference values:** for the tea-tasting table, OR and p were checked against the values I recall from R's `fisher.test` output (UNVERIFIED as quoted). The exact CI (0.2117355954, 626.2435305888) was solved independently in this session and matches scipy to 10 digits. The R CI I recalled (0.2117329, 621.9337505) differs, plausibly through `uniroot`'s default tolerance.
- **Two different contrasts:** the logistic regression compares contaminated vs **not** contaminated (clean + partial); Fisher compares contaminated vs **clean**. They are not the same contrast.
- **Not run:** LaTeX compilation (no pdflatex), and real-data behaviour (no real results exist yet).
