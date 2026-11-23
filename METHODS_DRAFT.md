# Methods (draft, written from the code as it runs on 2026-10-29)

Numbers that depend on real runs are marked **[TO FILL: source file]**. File and function names are given so each statement can be checked. Citations: see CITATIONS_TO_VERIFY.md.

## 1. Benchmarks and sampling (`common.py`)
- **GSM8K** (`openai/gsm8k`, `main`, test split), **MMLU** (`cais/mmlu`, `all`, test split), **HumanEval** (`openai/openai_humaneval`, test split).
- **Sampling:** 1,000 items per benchmark (`--limit 1000`), drawn without replacement from the dataset row indices with `random.Random(0).sample` (`--seed 0`) and sorted. HumanEval has 164 items, so all are used.
- **Item ids** are dataset row indices. The same (limit, seed) is used by every script, so ids align across overlap, evaluation and likelihood files.

## 2. Training-data overlap (`overlap.py`)
- **Corpora:**
  - Pile, infini-gram index `v4_piletrain_llama`, for Pythia.
  - Dolma v1.7, index `v4_dolma-v1_7_llama`, for OLMo-1B-0724-hf.
  - Every query is a `count` query; a window "occurs" if its count is ≥ 1.
- **Text queried:** only the item text: the GSM8K question, the MMLU question **without** its answer options, and the HumanEval prompt (signature + docstring, without the canonical solution or tests).
- **GSM8K and MMLU windows:** contiguous windows of 13 whitespace-separated words, re-joined with single spaces.
  - Items with fewer than 13 words are queried whole if they have at least 10 words; otherwise they get no window and their overlap is missing (excluded downstream).
  - Of all windows of an item, at most 15 are used, evenly spaced along the item: index `round(i·(n−1)/14)`.
- **HumanEval windows:** 32 whitespace tokens, sent as the exact source slice (newlines and indentation kept); minimum 16 tokens; same 15-window subsampling.
  - **Long-query fallback:** if the API rejects a window (HTTP 400/413/414/422), the window is replaced by its first half, never fewer than 16 tokens, and shortened again on a further rejection.
  - Each replacement is logged. The fallback rate is **[TO FILL: overlap_fallback_summary.csv]**.
- **Overlap score** of an item = fraction of its windows with count ≥ 1. `max_count` = largest count.
  - Windows whose query failed after 6 attempts (exponential backoff; HTTP 403/429/5xx and network errors retried) are excluded from the denominator, never counted as zero.
  - overlap.csv is written only when every benchmark × corpus pair has all its planned windows (`--allow-partial` otherwise marks rows `complete=False`, and analysis excludes them).
  - Completeness at the time of the final run: **[TO FILL: overlap_run.log, overlap_partial_notice.txt if any]**.
- **Strata** (per the corpus a model was trained on; `analyze.assign_stratum`):
  - clean: overlap = 0
  - partial: 0 < overlap < 0.5
  - contaminated: overlap ≥ 0.5 (`--thr 0.5`)
- **Items per stratum:** **[TO FILL: strata_n.csv]**. Items without windows: **[TO FILL: coverage.csv]**.

## 3. Perturbations
- **MMLU** (`evaluate.permute_choices`): the four options are shuffled with an RNG seeded by (seed, item id); an identity shuffle is replaced by a rotation. Items whose options contain "above", "both", "neither", "none of" or "all of" are not permuted. Coverage: **[TO FILL: coverage.csv items_perturbable]**.
- **GSM8K** (`perturb_gsm8k.py`): the reference solution's calculator annotations become a program.
  - Each numeric literal must resolve uniquely to a question number, an earlier step, or a fixed constant (number words, non-integer/ordinal/percentage/time numerals, common unit constants such as 7, 12, 60, 100); otherwise the item is skipped with a recorded reason.
  - Each question number v is replaced by a distinct integer in [⌈v/2⌉, 2v]. An assignment is accepted only if integer steps stay integers, signs are preserved, and the final answer changes (≤ 200 tries per item).
  - The gold answer is recomputed by the program. A re-check (`--validate`) re-evaluates 100 random records independently.
  - Coverage: **[TO FILL: perturbed_gsm8k.jsonl rows / 1000; skip reasons in perturbed_gsm8k_skipped.csv]**.
- **HumanEval** (`perturb_humaneval.py`): AST-based renaming of function parameters and local variables to fresh names, formatting preserved.
  - Builtins, imports, module-level names, globals/nonlocals, class bodies and dunders are never renamed. Keyword arguments in tests are renamed with the callee's parameters.
  - An item is kept only if the original solution passes the original tests, the renamed solution passes the adapted tests, and the renamed AST equals the original under the rename map.
  - Coverage: **[TO FILL: perturbed_humaneval.jsonl rows / 164]**.
- **Manual check:** 50 GSM8K and 30 HumanEval perturbed items, verdict correct / wrong / unclear. Fraction correct with a Wilson 95% CI: **[TO FILL: perturb_check_*_score.csv]**. The renamed canonical solutions are also run against the **original** tests: **[TO FILL: canonical_pass_rate]**.

## 4. Models and evaluation protocol (`evaluate.py`, `models.json`)
- **Models:**
  - Pythia-410M, 1B and 1.4B (Pile), and OLMo-1B-0724-hf (Dolma v1.7): float32, the primary models.
  - Pythia-2.8B: float16 only, as a labelled sensitivity model, analysed separately (`--include-sensitivity`, `*_sensitivity.csv`).
  - Qwen2.5-1.5B (float32): a difficulty-control reference, not stratified.
- **Precision rules:** bfloat16 is refused for Pythia. Every output row records the dtype in the model name (`@fp32` / `@fp16`). Rows of different dtypes are never analysed together. Weights are loaded with Hugging Face `transformers` **[TO FILL: version, requirements.lock.txt]** on one RTX 5070 Laptop GPU (8 GB).
- **MMLU:**
  - Prompt: 5-shot, using the first 5 dev-split examples of the item's subject, in the format "question / A.–D. options / Answer:", left-truncated to 2,040 tokens.
  - Prediction: the argmax of the next-token logits over the tokens " A", " B", " C", " D".
- **GSM8K:**
  - Prompt: 5-shot, using the first 5 training items (calculator annotations removed), "Question: … / Answer: …".
  - Decoding: greedy, at most 256 new tokens; output cut at the next "Question:".
  - Scoring: the prediction is the number after "####" if present, else the last number. Correct if it is within 10⁻⁴ of the gold answer.
- **HumanEval:**
  - Prompt and decoding: zero-shot, greedy, at most 512 new tokens, stopping at "\ndef ", "\nclass ", "\nif __name__" or "\nprint(".
  - Scoring: pass@1, with tests run in a subprocess with a 10 s timeout.
- **Perturbed variants** are scored with the same protocol.
- **GPU safety:** a run is refused if the weights alone exceed 90% of free GPU memory. A CUDA out-of-memory error during generation is retried once with half the token budget; a second failure skips that item-variant (no row, logged). Counts: **[TO FILL: generation_skips.csv]**. Peak memory: **[TO FILL: gpu_memory_log.csv]**.

## 5. Likelihood and Min-K% scoring (`itemloglik.py`)
- **Texts scored per item:**
  - the item text alone;
  - item text + "\n" + gold answer, where the answer is the full GSM8K reference solution, the MMLU correct option text, or (for HumanEval) the prompt directly followed by the canonical solution.
- **Tokenization:** a BOS token (EOS if there is no BOS) is prepended so every text token is scored. Texts are truncated to the model's context length.
- **Scores:**
  - mean token log-probability (natural log);
  - Min-K% Prob: the mean of the lowest 20% of token log-probabilities (k = 0.2, at least one token).
- **Computation:** forward passes are batched (8, right padding masked); log-softmax is always computed in float32.

## 6. Statistics (`analyze.py`)
- **Accuracy per stratum:** per model × benchmark, with percentile bootstrap 95% CIs (1,000 resamples, seed 0).
- **Contamination sensitivity:** paired raw − perturbed correctness per item, per stratum, with bootstrap CIs.
- **Logistic regression:** `correct ~ contaminated + log(n_words) [+ ref_correct]` per model × benchmark. Reported: odds ratio with Wald 95% CI.
  - Pairs with only one stratum or a constant outcome are not fitted; the reason is written to regression.csv.
  - **Not estimable:** a fit counts as not converged if the optimiser did not converge, the outcome is constant within the contaminated or the non-contaminated items (perfect separation), any standard error is infinite or missing, or |log OR| > 15. Its OR, CI and p are left empty and reported as "not estimable" with the reason. Count: **[TO FILL: regression.csv converged]**.
  - **Fisher exact test:** contaminated vs clean items (partial items excluded), reported for every pair with both groups non-empty. Conditional maximum-likelihood OR, exact 95% CI, two-sided p.
  - **Holm / BH** are applied to the converged logistic fits in regression.csv (`p_holm`, `p_bh`).
  - Clean-only corpus × benchmark pairs are reported as negative controls.
- **Contamination-adjusted accuracy** = accuracy on clean items; inflation = raw − adjusted. Ranking stability: Kendall τ between raw and adjusted model rankings per benchmark, computed when ≥ 3 models are available.
- **Likelihood analysis:**
  - Per stratum: means with bootstrap CIs, and Spearman ρ between score and overlap.
  - OLS with HC3 robust SEs: `score ~ contaminated + log(n_words) [+ reference model's score on the same text]` per model × benchmark × score.
- **Multiple comparisons:** Holm and Benjamini–Hochberg adjusted p-values (`p_holm`, `p_bh`) over all tests in `loglik_regression.csv`; raw p-values are reported alongside.
- **Primary outcome:**
  - If the upper 95% CI bound of overall accuracy is ≤ chance + 0.05 (chance: MMLU 0.25, GSM8K and HumanEval 0), accuracy is treated as descriptive and likelihood is the primary outcome (`primary_outcome` column).
  - Strata with n < 30 are flagged (`strata_n.csv`).

## 7. Validation of the overlap flag (`validate_sample.py`)
- **Sample:** 200 items with overlap > 0, stratified equally over partial and contaminated (leftover quota reassigned), shuffled, seed 0.
- **Snippet rule:** the same for every item. The item's highest-count window (ties: first window) is sent to infini-gram `search_docs` (one document, 80-token display).
- **Annotation:** blind; the sheet has no stratum, overlap score or counts.
  - yes = the snippet contains the question or answer near-verbatim
  - no = generic or boilerplate phrase only
  - unsure = excluded and counted
- **Dropped items:** items whose snippet could not be retrieved are dropped and counted. **[TO FILL: validation_key_summary.json]**
- **Reported:** precision of the flag per stratum, with a Wilson 95% CI. **[TO FILL: validation_precision.csv]**. Recall on clean items is not estimated.

## 8. Reproducibility
Every data-producing script appends its arguments, seed, limit, package versions, device, dtype and input SHA-256 to `run_manifest.jsonl`. Exact package versions are in `requirements.lock.txt`. Code and tests: **[TO FILL: repository / archive]**.
