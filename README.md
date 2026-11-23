# Contamination-Adjusted Evaluation

Code for the paper *"Contamination-Adjusted Evaluation: Quantifying How Training-Data Overlap Inflates LLM Reasoning Benchmark Scores"*.

The pipeline measures n-gram overlap between benchmark items (GSM8K, MMLU, HumanEval; 1,000 sampled items each, all 164 for HumanEval, seed 0) and each model's pretraining corpus (Pile for Pythia, Dolma 1.7 for OLMo-1B-0724) through the infini-gram API. It then compares model accuracy and per-item likelihood across clean, partial and contaminated items, with Qwen2.5-1.5B as a difficulty-control reference. Perturbed variants (numbers changed in GSM8K, answer options permuted in MMLU, identifiers renamed in HumanEval) measure contamination sensitivity.

**Precision rules.** Every output row names its model `<model>@<dtype>`, for example `EleutherAI/pythia-1.4b@fp32`. All Pythia models and OLMo run in float32; pythia-2.8b runs only in float16 as a labelled sensitivity model, analysed separately. bfloat16 is refused for Pythia. Rows of different dtypes are never pooled. For near-chance models, likelihood (mean log-prob, Min-K%) is the primary outcome and accuracy is descriptive.

## Run order

All commands run from this folder. Network = infini-gram API (api.infini-gram.io) or Hugging Face downloads. "You" = needs your action or judgement.

| # | Step | Command | Duration | Hardware | Network | You |
|---|---|---|---|---|---|---|
| 0 | Power: plugged in, Sleep = Never. Disk: ~19.8 GB of model weights (see Storage). | Windows settings | 5 min | – | – | **yes** |
| 1 | API preflight | `python preflight_api.py` | < 1 min | CPU | **API** | **yes** (your terminal) |
| 2 | Check the overlap script | `python overlap.py --help` (shows `--workers`) | seconds | CPU | – | – |
| 3 | Overlap, with restarts | `run_overlap.bat`; progress: `python overlap.py ... --status` or `python check_overlap_health.py` | ≈ 9 h for the ~24,200 remaining windows at the last measured ~44/min (UNVERIFIED with 2 workers) | CPU | **API** | start / stop |
| 4 | Perturb + evaluate + likelihood, all models in models.json (can run in parallel with 3) | `run_gpu_stages.bat` | UNVERIFIED: ≈ 1.6 h for pythia-410m (extrapolated from the `--limit 5` measurement), roughly 20–30 h for all six models | **GPU** | **HF** (first use of each model) | – |
| 5 | Analysis + figures + tables (after overlap.csv exists and is complete) | `run_analysis.bat` | minutes | CPU | – | – |
| 6 | Validation of the overlap flag | `python validate_sample.py export --n 200` → annotate **blind** → `python validate_sample.py score --filled validation_sample.csv --key validation_key.csv` | export ≈ 1–2 h (search_docs 10–30 s per call); annotation: your time | CPU | **API** (export) | **yes** |
| 7 | Perturbation checks | `python check_perturbed_sample.py export-gsm8k` and `export-humaneval` → verdicts → `score --filled ...` | export minutes; verdicts: your time | CPU | – | **yes** |
| 8 | Review | `coverage.csv`, `strata_n.csv` (flag n < 30), `gpu_memory_log.csv`, `generation_skips.csv`, `overlap_partial_notice.txt` (if any) | – | – | – | **yes** |

Details for steps 1–3: PREFLIGHT.md. Current status and open items: STATUS_FINAL.md.

## Storage

Hugging Face downloads model weights into `%USERPROFILE%\.cache\huggingface` (or `HF_HOME` if you set it). Sizes from the Hub, 2026-10-29: pythia-410m 0.91 GB (downloaded), pythia-1b 2.09, pythia-1.4b 2.93, pythia-2.8b 5.68, OLMo-1B-0724-hf 5.12, Qwen2.5-1.5B 3.09 GB; 18.9 GB still to download. Choose where the cache goes before step 4.

`python check_storage.py` (read-only, downloads nothing) prints the cache path and size, free space, and PASS/FAIL per model with a 3 GB safety margin. `run_gpu_stages.bat` runs it first and **skips** every model whose download would not fit, with a visible message. `evaluate.py` / `itemloglik.py --cache-dir <folder>` point a single run at another cache. The project folder is inside OneDrive: see ONEDRIVE_NOTES.md before long runs.

Paper drafts written from the code: METHODS_DRAFT.md, LIMITATIONS_DRAFT.md, RESULTS_SKELETON.md, CITATIONS_TO_VERIFY.md ([TO FILL] / UNVERIFIED marked).

## Scripts and outputs (all outputs land in this folder unless noted)

| Script | Writes |
|---|---|
| `overlap.py` (via `run_overlap.bat`) | `overlap.csv` (written only when every benchmark x corpus is complete, unless `--allow-partial`, which appends the column `complete` and writes `overlap_partial_notice.txt`), `overlap_cache.json`, `overlap_failed.log`, `overlap_fallback.json/.log`, `overlap_fallback_summary.csv`, `overlap_run.log` |
| `preflight_api.py` | nothing (prints; exit code 1 on failure) |
| `perturb_gsm8k.py`, `perturb_humaneval.py` | `perturbed_gsm8k.jsonl` (+ `_skipped.csv`), `perturbed_humaneval.jsonl` |
| `evaluate.py` | `results.csv` (model, benchmark, id, variant, correct), `generation_skips.csv` (CUDA OOM retries / skips; skipped items have no row), `gpu_memory_log.csv` |
| `itemloglik.py` | `itemloglik.csv`, `gpu_memory_log.csv` |
| `analyze.py` (via `run_analysis.bat`) | `strata_n.csv`, `strata_accuracy.csv` (+ `primary_outcome`), `sensitivity.csv`, `regression.csv` (+ `skipped_reason`), `adjusted_scores.csv`, `loglik_strata.csv`, `loglik_regression.csv` (+ `p_holm`, `p_bh`), `coverage.csv`; with `--include-sensitivity` also `*_sensitivity.csv` |
| `report.py` | `paper_outputs\` (figures PDF+PNG, LaTeX tables), `paper_outputs\sensitivity\` |
| `texcheck.py` | nothing (structural check of the .tex files; pdflatex is not installed, so compilation is unverified) |
| `validate_sample.py` | `validation_sample.csv` (blind sheet with rubric), `validation_key.csv`, `validation_key_summary.json`, `validation_snippets.json` (cache), `validation_snippets_failed.log`, `validation_precision.csv` |
| `check_perturbed_sample.py` | `perturb_check_gsm8k.csv`, `perturb_check_humaneval.csv`, `*_score.csv` |
| every data-producing script | appends one line to `run_manifest.jsonl` (args, seed, limit, versions, device, dtype, input SHA-256) |

Supporting modules: `common.py` (loaders), `execution.py` (HumanEval test runner; not a security sandbox), `modelcfg.py` (model names, models.json), `gpumem.py` (GPU memory safety), `manifest.py`. Exact package versions: `requirements.lock.txt`; install with `requirements.txt` (CUDA torch build noted there).

## models.json

`{model: {"corpus": "pile"|"dolma"|null, "role": "primary"|"reference"|"sensitivity", "dtype": "float32"|"float16", optional "max_new_tokens_gsm8k", "batch_size", "fits_8gb"}}`. A model with `"fits_8gb": false` is skipped by `run_gpu_stages.bat` with a visible message.

## Tests

`python -m pytest tests -q`: offline (HTTP mocked, local fake servers), uses the GPU when available, never touches `overlap_cache.json` or `overlap.csv` in this folder. A synthetic end-to-end run lives in `_work\smoke\e2e\run_e2e.py`.
