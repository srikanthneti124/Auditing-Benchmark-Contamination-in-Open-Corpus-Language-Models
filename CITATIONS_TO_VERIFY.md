# Citations to verify

Every entry below is **UNVERIFIED**. Names are given as I recall them, for you to look up. No authors, years, venues or DOIs are given on purpose. Check each title and claim against the source before citing.

| # | Work / tool (name to look up) | Claim it supports in the Methods |
|---|---|---|
| 1 | infini-gram (paper "Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens", and its API documentation) | n-gram counts and `search_docs` over the Pile and Dolma indices; index names `v4_piletrain_llama`, `v4_dolma-v1_7_llama` |
| 2 | The Pile ("The Pile: An 800GB Dataset of Diverse Text for Language Modeling") | Pythia's pretraining corpus |
| 3 | Dolma ("Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research"), plus the v1.7 release notes | OLMo-1B-0724's pretraining corpus, and that it is v1.7 |
| 4 | Pythia ("Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling") | Models trained on the Pile in a fixed order; sizes used |
| 5 | OLMo ("OLMo: Accelerating the Science of Language Models") and the OLMo-1B-0724-hf model card | Open model trained on Dolma; the 0724 version uses Dolma v1.7 |
| 6 | Qwen2.5 technical report and the Qwen2.5-1.5B model card | Reference model; training data not disclosed |
| 7 | GSM8K ("Training Verifiers to Solve Math Word Problems") | Benchmark, calculator annotations, "####" answer format |
| 8 | MMLU ("Measuring Massive Multitask Language Understanding") | Benchmark; 5-shot dev-split protocol |
| 9 | HumanEval ("Evaluating Large Language Models Trained on Code") | Benchmark and pass@1 |
| 10 | Min-K% Prob ("Detecting Pretraining Data from Large Language Models") | Min-K% likelihood score as a membership signal (k = 20%) |
| 11 | Memorisation in LMs (e.g. "Quantifying Memorization Across Neural Language Models") | Verbatim memorisation increases with duplication and model size |
| 12 | N-gram contamination analysis in model reports (e.g. the 13-gram overlap analysis in the GPT-3 paper "Language Models are Few-Shot Learners") | Precedent for n-gram window overlap as a contamination flag |
| 13 | Perturbation-based contamination checks (e.g. "A Careful Examination of Large Language Model Performance on Grade School Arithmetic", GSM1k) | Performance drops on perturbed or fresh variants as evidence of contamination |
| 14 | Surveys of benchmark data contamination (find one current survey) | Framing: n-gram detection misses paraphrases; other detection approaches |
| 15 | Wilson score interval | CI for validation precision and perturbation checks |
| 16 | Holm step-down procedure | Family-wise error correction |
| 17 | Benjamini–Hochberg procedure | False-discovery-rate correction |
| 18 | HC3 heteroskedasticity-consistent standard errors | Robust SEs in the likelihood OLS |
| 19 | Percentile bootstrap | CIs for accuracy and likelihood means |
| 20 | Kendall's τ | Ranking stability between raw and adjusted scores |
| 21 | Software: Hugging Face `transformers` and `datasets`, PyTorch, statsmodels, SciPy, pandas (versions in `requirements.lock.txt`) | Implementation |
