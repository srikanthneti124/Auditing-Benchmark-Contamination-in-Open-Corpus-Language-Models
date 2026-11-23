# Limitations (draft)

Items 1–8 were listed by the author. Items 9 onward were found in the code (file references given). **[TO FILL]** marks numbers that come from real runs.

## Listed
1. **N-gram overlap misses paraphrased contamination.** Detection is exact matching of 13-word (text) or 32-token (code) windows against the corpus index. Reworded, translated or reformatted copies of an item are not detected, so "clean" means "no verbatim window found", not "unseen".
2. **GSM8K × Pile is a negative control, not a test.** In the partial cache (2026-10-29), 892 of 893 fully cached GSM8K items had overlap 0 against the Pile and none reached the 0.5 threshold. The Pythia contamination effect on GSM8K is therefore not estimable (final numbers: **[TO FILL: strata_n.csv]**).
3. **Perturbation coverage and selection bias.** GSM8K perturbation applies only where the calculator annotations can be resolved unambiguously (about a third of items in earlier tests; final **[TO FILL: coverage.csv]**). MMLU items with "all/none of the above"-type options are not permuted. HumanEval items with unsafe renaming patterns are skipped. Perturbable items are a non-random subset, so sensitivity estimates may not generalise to all items.
4. **The reference model may be contaminated.** Qwen2.5-1.5B is the difficulty control, but its training data are undisclosed. If it saw the same benchmarks, controlling for its score can absorb part of the contamination effect.
5. **Near-chance models support only the likelihood outcome.** Where accuracy's 95% CI upper bound is within 0.05 of chance, accuracy differences between strata are uninformative and only likelihood shifts are interpreted (`primary_outcome`). Likelihood shifts show familiarity with the text, not inflated benchmark scores.
6. **Multiple comparisons.** Holm and BH corrections are applied within `loglik_regression.csv` only. Accuracy regressions, per-stratum CIs, Spearman correlations and Kendall τ are uncorrected.
7. **Single seed.** Item sampling, perturbations and the validation sample all use seed 0. Generation is greedy. Variation over sampling seeds is not estimated.
8. **No LaTeX compile check.** pdflatex is not installed. The tables pass a structural check (`texcheck.py`) only.

## Found in the code
9. **Only the item text is queried.** `common.py` sets `text` to the GSM8K question, the MMLU question without its options, and the HumanEval prompt. Contamination of answers, options, solutions or tests is not measured.
10. **Window subsampling.** At most 15 evenly spaced windows per item (`overlap._subsample`), so the overlap score of long items is an estimate, not an exhaustive count.
11. **Short items are excluded.** Items with fewer than 10 words (16 tokens for code) get no window and missing overlap. In the 1,000-item MMLU sample this was 165 items. The analysed set is biased toward longer items.
12. **Whitespace normalisation.** Text windows are re-joined with single spaces (newlines lost), and matches are exact. Corpus copies that differ in punctuation, spacing or casing are missed. For HumanEval, windows that the API rejected are replaced by shorter 16-token windows (rate **[TO FILL: overlap_fallback_summary.csv]**), which match generic code more easily.
13. **Fixed threshold.** The contaminated stratum is overlap ≥ 0.5. No sensitivity analysis over the threshold is implemented (`--thr` exists but is run once).
14. **Corpus–model match.** The Dolma v1.7 index is matched to OLMo-1B-0724-hf by name. Whether the index covers OLMo's full training mix was not verified. Counts come from Llama-tokenized indices, whatever tokenizer each model uses.
15. **Prompt design.** Few-shot exemplars are fixed (the first 5 GSM8K training items; the first 5 MMLU dev items per subject) and may themselves be memorised. There is one prompt format and no prompt variation.
16. **Scoring heuristics.** GSM8K takes the number after "####", else the last number in the output. MMLU uses next-token letter logits without calibration. HumanEval is greedy pass@1 only. HumanEval tests run in a subprocess with a timeout; this is not a security sandbox.
17. **Likelihood scales differ across models.** Log-probabilities and Min-K% depend on each model's tokenizer, so likelihood effects are compared within a model, not across models. Texts longer than the context window are truncated.
18. **Precision is confounded with size.** Only Pythia-2.8B runs in float16 (memory limit). Its differences from the float32 models mix scale and precision; it is reported separately as a sensitivity model.
19. **OOM handling changes the protocol for affected items.** After a CUDA out-of-memory error, an item is regenerated with half the token budget; after a second error it is skipped (**[TO FILL: generation_skips.csv]**). A skipped item makes model id sets differ, and analysis stops until it is rerun.
20. **Small strata and CIs.** Percentile bootstrap CIs and Wald CIs are unreliable for small strata (n < 30 is flagged in `strata_n.csv`). Logistic regressions can fail or be unstable under separation.
21. **Validation scope.** The validation measures only the precision of the overlap flag, with one annotator (no inter-rater agreement) and one snippet per item, from one document. Items whose snippet could not be retrieved are dropped (**[TO FILL: validation_key_summary.json]**).
22. **Earlier overlap failures.** About 107 GSM8K × Pile windows were lost without a log line by an earlier version of the overlap job. The current version retries them and refuses to write overlap.csv while any are missing (UNVERIFIED until the final run).
