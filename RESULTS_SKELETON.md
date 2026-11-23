# Results (skeleton; no numbers until the real runs)

Every placeholder names the file that fills it. Tables and figures come from `paper_outputs\` (`report.py`); the sensitivity model's versions are in `paper_outputs\sensitivity\`.

## R1. Coverage and overlap
- Items sampled, with overlap computed per corpus, evaluated, perturbable, and scored for likelihood: **Table: `tab_coverage.tex`** (from `coverage.csv`). [TO FILL]
- Overlap completeness and HumanEval fallback rate: [TO FILL: `overlap_run.log` final table, `overlap_fallback_summary.csv`]
- Items per stratum per model × benchmark, with n < 30 flags: [TO FILL: `strata_n.csv`] (suggested appendix table)
- Clean-only negative controls (expected: GSM8K × Pile): [TO FILL: printed by `analyze.py`; confirm against `strata_n.csv`]

## R2. Validity of the overlap flag
- Precision per stratum and overall (Wilson 95% CI), unsure and dropped counts: [TO FILL: `validation_precision.csv`, `validation_key_summary.json`]
- Perturbation checks: fraction judged correct (Wilson CI) and the canonical-solution pass rate: [TO FILL: `perturb_check_gsm8k_score.csv`, `perturb_check_humaneval_score.csv`]

## R3. Which outcome each model supports
- **Table: `tab_outcome_support.tex`** (accuracy, CI upper bound, chance, near-chance flag, primary outcome). [TO FILL]
- One sentence per benchmark on which models are near chance: [TO FILL]

## R4. Accuracy by contamination stratum (models with primary outcome = accuracy)
- **Figure: `fig_accuracy_by_stratum.pdf`**; **Table: `tab_strata_accuracy.tex`**. [TO FILL]
- Logistic regression odds ratios: **Figure: `fig_odds_ratio_forest.pdf`**; **Table: `tab_regression.tex`** (including the "Not fitted" reasons). [TO FILL]

## R5. Contamination-adjusted scores
- **Figure: `fig_raw_vs_adjusted.pdf`**; **Table: `tab_adjusted_scores.tex`** (raw, clean-subset, inflation). [TO FILL]
- Ranking stability: Kendall τ per benchmark, printed by `analyze.py` (≥ 3 models only). [TO FILL]

## R6. Sensitivity to perturbation
- **Figure: `fig_sensitivity.pdf`**; **Table: `tab_sensitivity.tex`** (raw − perturbed per stratum). [TO FILL]

## R7. Likelihood signals (primary outcome for near-chance models)
- **Figure: `fig_loglik_effect.pdf`**; **Table: `tab_loglik_regression.tex`** (coefficient, CI, p, p Holm, p BH). [TO FILL]
- Per-stratum means and Spearman ρ with overlap: [TO FILL: `loglik_strata.csv`] (appendix)

## R8. Precision sensitivity analysis (Pythia-2.8B, float16)
- Same figures and tables in `paper_outputs\sensitivity\`, reported separately and never pooled with float32 models. [TO FILL]

## R9. Compute notes
- Peak GPU memory and seconds per item per model × benchmark: [TO FILL: `gpu_memory_log.csv`]
- Items skipped after out-of-memory: [TO FILL: `generation_skips.csv`]
