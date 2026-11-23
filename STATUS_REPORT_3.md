# STATUS_REPORT_3: read-only audit (2026-11-01 05:13–05:40)

**Verdict: NOT COMPLETE.** Overlap and one model (pythia-410m@fp32) are done; the GPU stages stopped while starting pythia-1b, and no analysis, validation or perturbation check has run.

## 1. DONE
| Item | Evidence (this session) |
|---|---|
| Overlap complete | `overlap_run.log`: attempt 3 "remaining 0" for all 6 pairs, "saved 4328 rows -> overlap.csv", exit 0 at 08:08:26 on 29/10. `overlap_cache.json` has **48,424** entries (24,212 per index), SHA-256 `318d8fcd…f22e` (changed from `17cb80fd…b47c4` because overlap finished). No `complete` column, no `overlap_partial_notice.txt`. |
| 403 failures recovered | `overlap_failed.log`: 226 lines, 224 unique windows, all "HTTP 403" (first "HTTP 403: {"message":"Forbidden"}"), 05:09–08:07. **224/224 are now in the cache.** |
| Perturbations | `perturbed_gsm8k.jsonl` 343 rows (34.3%), `perturbed_humaneval.jsonl` 162 rows (162/164); MMLU 928/1000 permuted (results.csv). |
| pythia-410m@fp32 | results.csv: GSM8K 1000 raw / 343 perturbed; MMLU 1000 / 928; HumanEval 164 / 162. itemloglik.csv: 1000 / 1000 / 164. No duplicate keys; dtype suffix present. `gpu_memory_log.csv`: peak ≤ 3,873 MiB; 0 skipped items. |
| Ids aligned | results, itemloglik and overlap (both corpora) have identical id sets per benchmark, equal to the loader sample (limit 1000, seed 0). |
| Environment | torch 2.14.1+cu130, CUDA True, RTX 5070 Laptop GPU. HF_HOME `D:\hf_cache` (process and user). Free space: C: 11.07 GB, D: 143.54 GB. |

## 2. NOT DONE
| Item | What is missing | Blocking |
|---|---|---|
| GPU stages, 5 models | 0 rows for pythia-1b, pythia-1.4b, pythia-2.8b@fp16, OLMo-1B-0724-hf, Qwen2.5-1.5B (2,343 / 2,928 / 490 expected rows each for GSM8K / MMLU / HumanEval, including likelihood) | **yes** |
| Analysis and report | No strata_*, sensitivity, regression, adjusted_scores, coverage, loglik_* files, and no paper_outputs/ | yes (needs the GPU stages for multi-model claims) |
| Validation of the overlap flag | No validation_sample / key / precision files | yes (paper) |
| Perturbation checks | No perturb_check_* files | yes (paper) |

## 3. Why the GPU stages stopped (inference; the batch file writes no log, so console output is lost)
- **Timeline:**
  - pythia-410m itemloglik finished at 08:05:05 (`gpu_memory_log.csv`).
  - pythia-1b `model.safetensors` was fully downloaded at 08:05:40: 2,090,701,528 bytes, matching the Hub size; no `.incomplete` files.
  - `from_pretrained` reached its last lookup (`.no_exist/…/generation_config.json`) at 08:05:46.
  - **No `evaluate.py` manifest line for pythia-1b.** That line is written right after `load_model()` returns; for 410m it came 2 s after the download. So the process ended between the end of `from_pretrained` (then `.to("cuda")`) and `write_manifest`, about 08:05:46.
- **Checked and excluded** (Windows logs, 29/10 08:00–08:20):
  - No System events at all, so no shutdown or sleep at that time. Modern Standby began only at 10:59:13 and shutdown at 17:58:56.
  - No Resource-Exhaustion (2004) events. RAM is 15.7 GB; page file peak 2,135 of 10,620 MB.
  - No display-driver (4101/nvlddmkm) events.
  - No python crash in the Application log.
  - 0 Code Integrity blocks between 07:55 and 08:20.
  - The `LiveKernelEvent` 141/193 reports at 08:07 carry the same parameters as those at 05:11 and 11:14, which looks like re-submitted old reports (inference).
- **Candidates:**

| Candidate | For | Against |
|---|---|---|
| Download failure | – | file complete, sizes match |
| System RAM out of memory | – | no exhaustion event; page-file headroom |
| Shutdown / sleep | – | no event before 10:59 |
| GPU out of memory in `.to("cuda")` | stop point fits | 3.77 GiB weights vs about 6.8 GiB free measured earlier; the 410m process had already exited |
| Python exception (traceback only on screen) | the stop-on-error batch file would print the error and exit, leaving exactly this trace | no saved console output |
| Window closed / Ctrl+C | leaves exactly this trace; the overlap window restarted at 08:05:25 | no evidence either way |

- **Most likely:** a Python exception at model load / `.to("cuda")`, or the window being closed / interrupted. **I cannot distinguish them; UNVERIFIED.** No exact failing line exists on disk.
- **New since 29/10:** Windows Application Control blocked `…scipy\sparse\csgraph\_min_spanning_tree.cp312-win_amd64.pyd` at 05:13:53 today (Code Integrity 3077, policy `{0283ac0f-…}`, likely Smart App Control, UNVERIFIED). This interrupted my first pytest run at collection. Imports of scipy and all scripts **succeed on retry**, so the block is intermittent.

## 4. Where it stopped
- **Last stage completed:** itemloglik for pythia-410m@fp32 (08:05:05). Overlap finished separately at 08:08:26.
- **First stage not completed:** evaluate for pythia-1b@fp32, during model load at about 08:05:46.

## 5. Commands to finish (from cmd in the project folder; power plugged in, Sleep = Never, OneDrive paused)
| # | Command | Duration | Needs |
|---|---|---|---|
| 1 | Diagnose (shows any error): `set RUN_MANIFEST=_audit_tmp\probe_manifest.jsonl` then `python evaluate.py --model EleutherAI/pythia-1b --benchmarks mmlu --limit 5 --dtype float32 --out _audit_tmp\probe.csv --gpu-log _audit_tmp\probe_gpu.csv --skip-log _audit_tmp\probe_skips.csv` | ~2 min | GPU |
| 2 | `run_gpu_stages.bat > gpu_stages.log 2>&1` (watch: `powershell Get-Content gpu_stages.log -Wait`). 410m is skipped (`--skip-existing`); downloads about 16.8 GB to D:. | UNVERIFIED: 410m took ≈1.9 h for evaluate; 5 larger models ≈ 15–25 h | GPU, HF network |
| 3 | `run_analysis.bat` | minutes | CPU |
| 4 | `python validate_sample.py export --n 200` → annotate blind → `python validate_sample.py score --filled validation_sample.csv --key validation_key.csv` | 1–2 h + your time | API, you |
| 5 | `python check_perturbed_sample.py export-gsm8k` / `export-humaneval` → verdicts → `score --filled …` | minutes + your time | CPU, you |

## 6. Results sanity (pythia-410m@fp32, existing files only; bootstrap seed 0)
- **Accuracy:**
  - GSM8K 0.021 [0.013, 0.030]
  - HumanEval 0.012 [0.000, 0.031]
  - MMLU 0.268 [0.241, 0.296] (chance 0.25)
  - **All three are near chance by the analysis rule** (CI upper ≤ chance + 0.05), so likelihood is the primary outcome.
- **Pile strata:**
  - GSM8K: 999 clean, 1 partial, 0 contaminated.
  - HumanEval: 164 clean (no overlap at all).
  - MMLU: 693 clean, 31 partial, 111 contaminated, 165 without a window. **Only MMLU allows a contamination contrast for Pythia.**
  - MMLU accuracy: clean 0.264 (n = 693), contaminated 0.279 (n = 111).
  - MMLU mean log-prob of the question: clean −3.591, contaminated −3.418 (descriptive only; not length-adjusted, no CI or test).
- **Dolma strata** (for OLMo, which has no rows): GSM8K 63 contaminated, HumanEval 11, MMLU 269.
- **What one model can show:** at most a single-model, near-chance, likelihood-only, MMLU-only observation. Cross-model inflation, contamination-adjusted rankings (Kendall τ needs ≥ 3 models), the Dolma contrasts and the difficulty control all need OLMo, Qwen and the other Pythias.

## 7. Paper readiness
| Section | Rating | Reason |
|---|---|---|
| Methods | PARTIAL | METHODS_DRAFT.md matches the code. Coverage numbers can now be filled (34.3% / 92.8% / 98.8% perturbable; overlap complete; 165 MMLU items without a window). Model-run details are still missing. |
| Results | NOT READY | One model, near chance, no analysis outputs. |
| Limitations | PARTIAL | LIMITATIONS_DRAFT.md exists. Add: Pile × HumanEval has zero overlap; Pile × GSM8K has 0 contaminated items; 224 windows needed retries after 403s. |

## 8. Tests and UNVERIFIED
- **pytest:**
  - Run 1 was interrupted by the Application Control block (above).
  - Run 2: **628 passed, 3 failed**. The 3 failures were offline subprocess tests needing pythia-70m in the active cache (D:), where it was missing.
  - A later test in the same run **downloaded pythia-70m (166,029,852 bytes) into `D:\hf_cache` at 05:18:04**. This side effect was caused by the audit.
  - The 3 tests rerun: **3 passed**.
- **UNVERIFIED:**
  - The exact cause of the pythia-1b stop: verified by command 1 (and by logging step 2).
  - That the Code Integrity policy is Smart App Control: Windows Security → App & browser control.
  - Durations for the remaining models: `gpu_memory_log.csv` after step 2.
  - LaTeX compilation: pdflatex.
