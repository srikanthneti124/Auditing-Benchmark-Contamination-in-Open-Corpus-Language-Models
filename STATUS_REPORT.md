> **SUPERSEDED (2026-10-29).** Historical record of an earlier state; commands, defaults and file formats below may no longer be current. Current behaviour: README.md and PREFLIGHT.md; current status: STATUS_FINAL.md.

# STATUS_REPORT — Contamination-Adjusted Evaluation

Audit date: 2026-10-24, about 22:36 local time. Read-only: no project file was modified. This report is the only file created. Tests ran with `PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -p no:cacheprovider`, which writes no cache.

## 1. Verdict
The code is written and its unit tests pass. **No stage has produced a result yet.** The project folder holds source files, tests and one intermediate file, `overlap_cache.json`. Step 1, `overlap.py`, is **running now**: PID 28348, started 18:10, about 70% through the first of four benchmark×corpus passes. It writes no `overlap.csv` until all passes finish, and its command line does not include HumanEval. There are no perturbation files, no `results.csv`, no `itemloglik.csv`, no analysis CSVs, no figures and no validation sample. No headline result exists. The data collected so far suggests Pythia×GSM8K will have **no contaminated stratum**: every one of 11,600 Pile windows has count 0.

## 2. Stage status
| Stage | Status | Evidence | Missing |
|---|---|---|---|
| overlap | **PARTIALLY RUN (in progress)** | `Win32_Process` shows PID 28348, `overlap.py --benchmarks gsm8k mmlu --corpora pile dolma --limit 1000`, started 18:10:17. `overlap_cache.json` is 1,067,324 B, modified 22:34:48. It has 11,600 entries, all `v4_piletrain_llama`, **0 nonzero**, max count 0. GSM8K×Pile: 695/1000 items fully cached, 105 missing exactly one window, 200 not started. GSM8K×Dolma, MMLU×Pile and MMLU×Dolma: 0 windows cached. | `overlap.csv`. Any Dolma counts. Any MMLU counts. HumanEval overlap (not in the running command). |
| perturbations | **WRITTEN BUT NEVER RUN** (in project) | No `perturbed_gsm8k.jsonl`, `*_skipped.csv` or `perturbed_humaneval.jsonl` in the folder. Development runs went to a scratch directory outside the project (343/1000 GSM8K, 162/164 HumanEval); those are not project artifacts. | Both jsonl files and the skip CSV. |
| evaluation | **WRITTEN BUT NEVER RUN** | No `results.csv` in the folder, or anywhere under Desktop/home to depth 3 (`find`). | All model runs. |
| itemloglik | **WRITTEN BUT NEVER RUN** | No `itemloglik.csv`. | All. |
| analysis | **WRITTEN BUT NEVER RUN** | No `strata_accuracy/sensitivity/regression/adjusted_scores.csv`. Its inputs do not exist. | All. |
| report | **WRITTEN BUT NEVER RUN** | No `paper_outputs/`. | All. |
| validation | **WRITTEN BUT NEVER RUN** | No `validation_sample.csv` or `validation_precision.csv`. | Export, annotation, scoring. |
| tests | **COMPLETE** | `479 passed in 17.06s`, no failures. | — |

No log files exist in the project folder. The `.log` files `find` turned up are unrelated (LaTeX logs under `Desktop/Job Resumes`).

## 3. Consistency checks
| Check | Result | Evidence |
|---|---|---|
| Same `--limit`/`--seed` id sets across overlap, results and itemloglik | UNVERIFIED | None of the three CSVs exists. The cache keys match windows from `load_benchmark('gsm8k',1000,0)` (695 items fully matched), so the running job uses limit=1000, seed=0. |
| No duplicate (model, benchmark, id, variant) rows | UNVERIFIED | No `results.csv`. |
| Every model maps to a corpus or is the reference | UNVERIFIED | No `results.csv`. |
| Positive control (index returns nonzero for known text) | **PASS** | curl `count` on "We the People of the United States, in Order to form a more perfect Union, establish Justice" against `v4_piletrain_llama` → `"count":57`. The all-zero GSM8K counts are not an index fault. |
| Pairs with only one stratum | **FAIL (provisional)** | GSM8K×Pile: 0 of 11,600 windows nonzero, so every item processed so far is "clean". If the remaining 200 items are also zero, Pythia×GSM8K has a single stratum and `analyze.py` skips its regression silently. Other pairs: UNVERIFIED. |
| n < 30 in any stratum | UNVERIFIED | No overlap.csv or results. Known now: **165/1000 MMLU items produce no windows** (questions under 10 words), so their overlap will be NaN and they will be dropped. |
| Cache integrity | Note | 105 GSM8K items each miss exactly one window, at random positions, in ordinary ASCII text. This is *consistent with* failed API calls (`Counter.count` returns None after 6 retries and the window is dropped, so `n_windows` is 14). The cause is UNVERIFIED; the process's stderr is not saved. |

## 4. Where the work stopped
- **Last completed stage:** tests only.
- **First incomplete stage:** overlap, which is in progress.
- **Pipeline stage not yet started:** everything after overlap.

## 5. Commands to finish (in order)
Rates are measured from the cache in this session: 7,400 entries at 21:00 and 11,600 at 22:34 is about 45 windows/min. Model run times are not measured and are given as UNVERIFIED estimates.

| # | Command | Expected time | Hardware |
|---|---|---|---|
| 0 | Let PID 28348 finish. Do **not** start another `overlap.py` meanwhile: both save to the same cache and the last writer wins. | About 33,000 unique windows remain, roughly 12 h at the measured rate, if the API does not rate-limit (it returned `Forbidden` to this machine at about 21:20). | network |
| 1 | `python overlap.py --benchmarks humaneval --corpora pile dolma --out overlap_humaneval.csv`, then concatenate with `overlap.csv` | 164 items × ≤15 windows × 2 corpora: about 1 h at the measured rate | network; needs a fix for Python TLS on this machine (requests fails `CERTIFICATE_VERIFY_FAILED`; e.g. `truststore`) |
| 2 | Install a CUDA build of torch. The current build is `2.14.1+cpu` and `cuda.is_available()` is False, although `nvidia-smi` shows an RTX 5070 Laptop, 8,151 MiB. | minutes | — |
| 3 | `python perturb_gsm8k.py --limit 1000 --seed 0 && python perturb_gsm8k.py --validate` | under 1 min (development run) | CPU |
| 4 | `python perturb_humaneval.py --seed 0` | about 15 s (development run) | CPU |
| 5 | `python evaluate.py --model <M> --benchmarks gsm8k mmlu humaneval --limit 1000` for each Pythia model, each OLMo model and the reference model | UNVERIFIED. On CPU, pythia-70m took 3 min for 20 items per benchmark, so 1B+ models on CPU are impractical. | **GPU**. OLMo-7B in bf16 is about 14 GB, larger than 8 GB VRAM; it needs a smaller model or quantization. |
| 6 | `python itemloglik.py --model <M> --benchmarks gsm8k mmlu humaneval --limit 1000` for each model including the reference | UNVERIFIED; one forward pass per text | **GPU** recommended |
| 7 | `python analyze.py --ref-model <ref> --loglik itemloglik.csv` | seconds | CPU |
| 8 | `python report.py` | seconds | CPU |
| 9 | `python validate_sample.py export --n 200`, annotate by hand, then `python validate_sample.py score --filled validation_sample.csv` | export: up to about 2 h (search_docs took 10–37 s per call); annotation is manual | network |

## 6. Paper readiness
| Section | Data exists today? |
|---|---|
| Methods | Partly. Pipeline design, windowing, perturbation rules and the statistical models are defined in code and tested. Final perturbation coverage numbers and overlap statistics do not exist yet. |
| Results | **No.** There is no accuracy, overlap stratum, sensitivity, regression or adjusted score. |
| Limitations | Partly. Known risks: Pile likely predates GSM8K (all zero counts so far), so Pythia may have no contaminated stratum; GSM8K perturbation covers about 34% of items (selection bias, development run); the reference model may itself be contaminated; many tests without multiple-comparison correction; annotation is unblinded; 165 MMLU items have no windows; GPU/VRAM limits on model choice. |

## 7. Not verified, and why
- **Overlap distributions, NaN counts, strata at thresholds 0 and 0.5:** `overlap.csv` does not exist yet.
- **All results.csv checks** (models, duplicates, id alignment, missing pairs), **accuracy and chance-level flags, contaminated-vs-clean contrasts, n per stratum:** no `results.csv`.
- **itemloglik coverage:** no file.
- **Freshness of analysis and report outputs:** none exist.
- **Cause of the 105 missing windows:** the running process's console output is not logged.
- **Whether the running job will finish, and whether Dolma and MMLU queries succeed:** they are future events, and the API has refused requests from this machine before.
- **Run times for model stages:** not measured on GPU.
