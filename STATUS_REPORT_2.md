> **SUPERSEDED (2026-10-29).** Historical record of an earlier state; commands, defaults and file formats below may no longer be current. Current behaviour: README.md and PREFLIGHT.md; current status: STATUS_FINAL.md.

# STATUS_REPORT_2 — read-only audit (2026-10-29 01:43 local)

**Verdict: NOT COMPLETE.** The overlap run stopped about half done on 2026-10-25 and nothing after it has run on real data. The code also breaks the dtype-labelling decision.

Audit scripts are in `_audit_tmp/` (`audit_overlap.py`, `prelim_strata.py`, `gpu.py`). The audit made no API calls and touched no process. Running pytest may have refreshed `.pytest_cache/`.

## 1. Inventory (full listing: `Get-ChildItem -Recurse`, this session)
- **Source:** overlap.py, perturb_gsm8k.py, perturb_humaneval.py, evaluate.py, itemloglik.py, analyze.py, report.py, validate_sample.py, common.py, execution.py, check_overlap_health.py, run_overlap.bat, run_after_overlap.bat, models.json, tests/ (8 files), plus six `.bak` files.
- **Data:** `overlap_cache.json` (2,234,963 B, modified 2026-10-25 03:39:42) and `overlap_cache.backup.json` (1,899,957 B, 02:17:56).
- **Test-only outputs:** `_work/smoke/*`, a 20-item offline chain on a **synthetic** `overlap_fixture.csv`. None of it is real data.
- **Logs:** none in the project root.
- **Expected but missing in the root:** overlap.csv, overlap_run.log, overlap_failed.log, perturbed_gsm8k.jsonl, perturbed_gsm8k_skipped.csv, perturbed_humaneval.jsonl, results.csv, itemloglik.csv, strata_accuracy.csv, strata_n.csv, sensitivity.csv, regression.csv, adjusted_scores.csv, loglik_strata.csv, loglik_regression.csv, paper_outputs/, validation_sample.csv, validation_precision.csv.

## 2. DONE
| Item | Evidence |
|---|---|
| Pipeline code written | Files listed above. |
| Unit tests pass | `python -m pytest tests -q` → **506 passed in 65.97s**, 0 failed. |
| Tests leave the live cache untouched | SHA-256 `17CB80FD…47C4` and mtime 03:39:42 were identical before and after pytest. |
| Test guard exists | `tests/test_overlap_parallel.py:119-121` (`_guard`) and `:102` (`chdir(tmp_path)`). It covers that file only; `tests/conftest.py` has no global guard. |
| Cache is valid and has not shrunk | 24,200 entries, parses. Backup has 20,600, all present in the live cache with identical values; the live cache has 3,600 more. |
| GSM8K × Pile overlap nearly done | 14,522/14,629 windows cached, 4 nonzero. 107 remain; these are the earlier silent failures. |
| run_overlap.bat includes HumanEval | `run_overlap.bat:22`: `--benchmarks gsm8k mmlu humaneval … --k 15 --workers 2`. |
| OLMo-0724 → Dolma 1.7; OLMo-1B-hf excluded | `models.json:6`; `overlap.py:49` maps `dolma` to `v4_dolma-v1_7_llama`. |
| Qwen used only as the reference | `models.json:7`. Analysis drops it from the model set: `analyze.py:138` and `:233`. |
| Near-chance rule in code | `report.py:202-214`: if the CI upper bound ≤ chance + 0.05, the primary outcome is likelihood. |
| fp32 passed by the handoff script | `run_after_overlap.bat:23` (`DTYPE=float32`); `:67` uses float16 only when the name matches "2.8b". |
| GPU works | torch 2.14.1+cu130, transformers 5.20.2, `cuda available True`, NVIDIA GeForce RTX 5070 Laptop GPU, Python 3.12.10. |
| Offline smoke chain ran | `_work/smoke/log_*.txt` and 6 figures + 7 tables, dated 2026-10-25 02:50–02:55. These are synthetic: plumbing only. |

## 3. NOT DONE
| Item | What is missing | Blocking? |
|---|---|---|
| Overlap | No overlap.csv. **24,224 of 48,424** planned unique windows remain: MMLU 7,693 × 2 corpora, HumanEval 1,890 × 2, GSM8K-Dolma 4,951, GSM8K-Pile 107. | yes |
| **Dtype in model name** | `evaluate.py:231` and `itemloglik.py:157` set `name = model + @revision` only. No `@fp32`/`@fp16` suffix exists; smoke rows read `EleutherAI/pythia-70m`. pythia-2.8b fp16 rows would be **pooled** with the fp32 rows in analysis, which has no dtype handling. | yes, the decision is violated |
| bf16 default | `evaluate.py:64-65`: `auto` means bf16 on GPU. Only the .bat prevents bf16 for Pythia; a manual run without `--dtype` would use bf16. | no (risk) |
| Perturbation (real) | No perturbed_*.jsonl in the root. | yes |
| Evaluation and likelihood | results.csv and itemloglik.csv are absent: 0 rows for every model × benchmark. | yes |
| Analysis, report, figures | Not produced. No n-per-stratum table exists for real data. | yes |
| Validation sample | Not exported, not annotated, no precision computed. | yes (paper) |
| Blind annotation | The export writes the `stratum`, `matched_window` and `corpus_snippet` columns to the annotation sheet (`validate_sample.py:132-134`). Nothing hides them. | yes, for a blinded claim |

## 4. BLOCKED
| Item | Blocker | Who must act |
|---|---|---|
| Finish overlap | The job is not running: no python processes at 01:43; the cache was last saved 2026-10-25 03:39. The API must be reachable from your terminal (PREFLIGHT step 2). | **you** run it |
| Dtype labelling and blind export | Code change needed in evaluate.py, itemloglik.py and validate_sample.py. This audit was read-only. | **assistant**, on your approval |
| Everything after overlap | `run_after_overlap.bat:30-33` refuses to start without overlap.csv. | follows from the overlap run |
| Annotation | Manual judgement. | **you** |

## 5. Where it stopped
- **Last stage completed:** overlap for GSM8K × Pile, apart from 107 windows.
- **First stage not completed:** the overlap run over all benchmarks.
- `run_overlap.bat` has never run: its redirect (line 37) would have created overlap_run.log, and that file does not exist.
- The cache grew 3,600 entries between the backup mtime (02:17:56) and the live mtime (03:39:42), about **44 entries/min**. This assumes the backup was taken at its mtime.
- The old PID 28348 is gone. Its exact command line and why it stopped are **UNVERIFIED**. The cache has 0 MMLU or HumanEval windows, and STATUS_REPORT.md:44 planned HumanEval as a separate run.

## 6. Commands to finish (in order)
| # | Command | Duration | Hardware |
|---|---|---|---|
| 0 | Approve the fix for dtype suffix, non-bf16 default and blind export; the assistant edits and reruns pytest | ~15 min | CPU |
| 1 | PREFLIGHT.md steps 2, 3 and 5 (API check; real-API test on a cache copy) | ~10 min | network |
| 2 | `run_overlap.bat`, monitored with `python check_overlap_health.py` (writes a small state file) | 24,224 windows ÷ 44/min ≈ **9 h**; faster with 2 workers if the API allows (UNVERIFIED) | network/CPU |
| 3 | Check that overlap_run.log ends with `saved … rows -> overlap.csv`, then inspect overlap_failed.log | min | CPU |
| 4 | `run_after_overlap.bat --dry-run`, then `run_after_overlap.bat` | UNVERIFIED; the smoke run took 115 s for 20 items × 3 benchmarks on pythia-70m, so expect many hours for 6 models × 1000 items | **GPU** |
| 5 | `python validate_sample.py export --n 200`, annotate blind, then `python validate_sample.py score --filled <file>` | ~2 h export + manual work | network |

## 7. Results sanity (Step 7)
- No results.csv or itemloglik.csv exists, so there is no raw accuracy and no contaminated-vs-clean contrast. **No headline result exists.**
- Preliminary strata from the cache (`_audit_tmp/prelim_strata.py`, fully cached GSM8K items only; overlap 0 = clean, ≥0.5 = contaminated):

| Pair | n | clean | partial | contaminated |
|---|---|---|---|---|
| GSM8K × Pile | 893 | 892 | 1 | **0** |
| GSM8K × Dolma | 630 | 572 | 12 | 46 |

- **GSM8K × Pile has only one stratum**, so `analyze.py:188` will silently skip the regression for every Pythia model on GSM8K. Pythia GSM8K contamination cannot be tested.
- MMLU and HumanEval strata are unknown. 165 of 1,000 MMLU items have no window (`audit_overlap.py`) and will get NaN overlap.

## 8. Paper readiness
| Section | Rating | Reason |
|---|---|---|
| Methods | PARTIAL | The design and code exist, but dtype labelling and blind annotation are not implemented as decided. Run parameters cannot be stated until the runs happen. |
| Results | NOT READY | No real overlap, evaluation, likelihood or analysis outputs. |
| Limitations | PARTIAL | Known items are in STATUS_UPDATE.md §8. Add: no contaminated GSM8K × Pile items; annotation not blinded under the current code. |

## 9. UNVERIFIED
| Claim | What would verify it |
|---|---|
| Old job's command, `--workers` use, and why it stopped | Process gone; Windows event log or your recollection. |
| API reachable now | PREFLIGHT step 2. |
| Remaining overlap time | Two `check_overlap_health.py` readings during the run. |
| fp32 fits for pythia-1.4b, OLMo and Qwen during generation; fp16 fits for 2.8b; OOM-halving path | The first real run. |
| Same ids across overlap, results and itemloglik | Code uses the same loader, `--limit 1000 --seed 0` (`common.py:11-15`, `run_after_overlap.bat:24-25`, `run_overlap.bat:22`). The real files do not exist to compare. |
| LaTeX compiles; real figures labelled | Run report.py on real data, then pdflatex. Only the synthetic smoke figures exist. |
| Per-pair n for MMLU, HumanEval and the full GSM8K × Dolma | overlap.csv, then `strata_n.csv` from analyze.py. |
