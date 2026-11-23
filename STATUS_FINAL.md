# STATUS_FINAL (2026-10-29)

**Verdict:** the coding is complete and tested offline. One item is **BLOCKED** on a decision only you can make: where to store about 19 GB of model weights (C: has about 4.6 GB free). Because of that, GPU measurements exist for pythia-410m only. Nothing here touched the API, the live cache, or any process except my own runs.

## 1. Phases
| Phase | Status | Evidence (this session) |
|---|---|---|
| 1 Validation protocol | **PASS** | Samples only overlap > 0. Snippets come from one rule (highest-count window, then search_docs). Blind sheet with the rubric in its header; key file read only by score. Snippet cache, retries, failure log. Dropped items are counted. `tests/test_validate_protocol.py` 16 passed. End-to-end (mocked): 34 kept, 6 dropped (`request_failed`); scope statement printed. |
| 2 Overlap robustness | **PASS** (real API UNVERIFIED) | `--status` accounting; refuses to write overlap.csv when incomplete (exit 3); `--allow-partial` adds a `complete` column plus a notice; analyze excludes `complete=False`; HumanEval long-query fallback (logged, rate in a CSV); `run_overlap.bat` restart loop (20 restarts, 60 s apart, log appended); `preflight_api.py`. `tests/test_overlap_robustness.py` 17 passed, including the real batch loop and a local fake server. End-to-end: complete, fallback 62/241 = 25.7%. |
| 3 GPU memory safety | **Code PASS; measurement BLOCKED** | Weights check (90% of free memory), OOM retry then skip (never scored, logged), `gpu_memory_log.csv`, `fits_8gb` skip. `tests/test_gpu_safety.py` 12 passed. Measured pythia-410m@fp32, `--limit 5`: peak 1730 MiB (GSM8K), 1718 MiB (HumanEval), 2237 MiB (likelihood, batch 8); 3.39 / 3.00 / 0.06 s per item for GSM8K / HumanEval / MMLU; 0 OOM skips. **Not measured:** pythia-1b, 1.4b, OLMo-1B-0724, because there is no disk space to download them. |
| 4 Perturbation checks | **PASS** | `check_perturbed_sample.py` (GSM8K 50, HumanEval 30 plus a programmatic run of the renamed solutions on the original tests). `tests/test_check_perturbed.py` 4 passed. End-to-end: 20/20 renamed canonical solutions pass the original tests. |
| 5 Additive statistics | **PASS** | `p_holm`/`p_bh` (family = all tests in the file), `primary_outcome`, `coverage.csv`, figure titles, `texcheck.py`. `tests/test_additive_stats.py` passed. Old analyze.py (pre-session backup) vs new on identical data: **all pre-existing columns identical in 14/14 files** (`_work/compare_analyze/compare.py`). |
| 6 Reproducibility | **PASS** | `run_manifest.jsonl` from 9 scripts (`tests/test_manifest.py` 4 passed); `requirements.lock.txt` (136 packages, `pip freeze`); README.md. |
| 7 Offline end-to-end | **PASS** | `_work/smoke/e2e/run_e2e.py`: 22 steps, exit 0, 662 s. All 10 figures opened: non-empty, labelled, near-chance panels marked. 20/20 .tex pass `texcheck.py`. Main outputs are byte-identical with sensitivity off and on. |

**Fixes along the way:**
- **End-to-end attempt 1** exposed a real bug: windows of 25–31 tokens had no fallback, so overlap.py correctly refused to write (exit 3). Fixed in overlap.py and covered by a regression test.
- **fig_sensitivity** lacked the † marking. Fixed.
- **Six tests** failed on first run because of mistakes in the tests themselves (fixtures, a Windows `\r\n`); all six corrected.

## 2. Files
- **New:**
  - `gpumem.py`: GPU memory safety
  - `manifest.py`: run manifest
  - `preflight_api.py`: API check
  - `check_perturbed_sample.py`: perturbation sheets
  - `texcheck.py`: LaTeX structural check
  - `README.md`
  - `requirements.lock.txt`
  - Six test files: `tests/test_validate_protocol.py`, `test_overlap_robustness.py`, `test_gpu_safety.py`, `test_check_perturbed.py`, `test_additive_stats.py`, `test_manifest.py`
- **Changed** (each backed up as `<name>.bak_20261029_HHMM`; diffs in `_work/diffs/`):
  - `evaluate.py`: OOM retry/skip, weights check, GPU log, `--gsm8k-max-new-tokens`, manifest
  - `itemloglik.py`: weights check, GPU log, manifest
  - `modelcfg.py`: run settings, `list-run`
  - `models.json`: pythia-410m measurement
  - `run_gpu_stages.bat`: per-model settings, `fits_8gb` skip
  - `validate_sample.py`: new protocol
  - `overlap.py`: accounting, refusal, `--allow-partial`, `--status`, fallback, manifest
  - `run_overlap.bat`: restart loop
  - `check_overlap_health.py`: counts fallbacks
  - `analyze.py`: incomplete exclusion, `primary_outcome`, Holm/BH, coverage, manifest
  - `report.py`: titles, coverage table, Holm/BH columns, manifest
  - `perturb_gsm8k.py`, `perturb_humaneval.py`: one manifest call each; logic unchanged
  - `PREFLIGHT.md`: rewritten
  - `requirements.txt`: pointer to the lock file
  - `STATUS_REPORT.md`, `STATUS_UPDATE.md`, `STATUS_REPORT_2.md`: "superseded" banner
- **Existing tests changed:**
  - `test_overlap_parallel.py::test_d_failed…`: one line adds `--allow-partial`. Phase 2.2 forbids writing an incomplete overlap.csv without it.
  - `test_dtype_roles_blinding.py`: 4 validation tests removed. They asserted the replaced protocol (all strata, no snippets); their intent (blind sheet, overwrite refusal, key-only stratum, key required) is re-tested in `test_validate_protocol.py`.

## 3. Tests and cache
`python -m pytest tests -q`: **597 passed, 0 failed** (531 at start). Live `overlap_cache.json` SHA-256 `17CB80FDBA17E0AFB4F583501893EC3A072059EA7A508B2EB71A4A69034B47C4`, mtime 2026-10-25 03:39:42, 2,234,963 bytes: **identical at start and end**. overlap.csv, results.csv and itemloglik.csv do not exist; no stray outputs in the project root.

## 4. UNVERIFIED
| Claim | What would verify it |
|---|---|
| API reachable; real count / search_docs responses | `python preflight_api.py` in your terminal |
| infini-gram rejects long queries with 400/413/414/422 (fallback trigger) | `overlap_fallback.log` / `overlap_failed.log` after the real run |
| Overlap time (~24,200 windows; ≈ 9 h at the old ~44/min) | Two `check_overlap_health.py` readings |
| pythia-1.4b@fp32 fits generation in 8 GB. Weights 5.27 GiB vs 6.83 GiB free (passes the 90% check); 410m peaked 0.7 GiB above its weights; expect about 6 GiB, which is tight | `run_gpu_stages.bat` → `gpu_memory_log.csv`, `generation_skips.csv` |
| 1b, OLMo, Qwen fp32 (5.75 GiB) and 2.8b fp16 (5.17 GiB) fit; full GPU run ≈ 20–30 h (extrapolated) | Same |
| OOM retry/skip on real hardware (simulated only) | `generation_skips.csv` |
| LaTeX compiles (structural check only; no pdflatex) | `pdflatex` on `paper_outputs/*.tex` |
| Ctrl+C → "Terminate batch job" inside the restart loop (loop tested via exit codes) | Press Ctrl+C once during `run_overlap.bat` |

## 5. Your handoff checklist (in order)
| # | Action | Duration | Needs |
|---|---|---|---|
| a | Plugged in; Sleep = Never. **Decide the model cache location**: free ≥ 20 GB on C:, or `setx HF_HOME <folder>` and open a new terminal | 5 min | you |
| b | `python preflight_api.py` (your terminal); expect `PREFLIGHT OK` | < 1 min | network |
| c | `python overlap.py --help` shows `--workers`, `--allow-partial`, `--status` | seconds | CPU |
| d | `run_overlap.bat`; monitor with `python check_overlap_health.py` or `overlap.py ... --status` | ≈ 9 h (UNVERIFIED) | network, CPU |
| e | `run_gpu_stages.bat --dry-run`, then `run_gpu_stages.bat` (may run in parallel with d) | ≈ 20–30 h (UNVERIFIED) | GPU, HF download ≈ 18.9 GB |
| f | `run_analysis.bat` once overlap.csv exists. The overlap run only writes it when complete; `--allow-partial` pairs are excluded by analyze | minutes | CPU |
| g | `python validate_sample.py export --n 200` → annotate `validation_sample.csv` without opening the key → `python validate_sample.py score --filled validation_sample.csv --key validation_key.csv` | 1–2 h export + your time | network, you |
| h | `python check_perturbed_sample.py export-gsm8k` and `export-humaneval` → fill `verdict` → `score --filled <file>` | minutes + your time | CPU, you |
| i | Review `coverage.csv`, `strata_n.csv` (flag n < 30), `gpu_memory_log.csv`, `generation_skips.csv` | 30 min | you |

## 6. Limitations affecting the paper's claims
- **GSM8K × Pile is clean-only** (892/893 cached items have overlap 0, 0 contaminated). Pythia GSM8K contamination cannot be estimated; analyze reports the pair as a negative control.
- **Near-chance models:** accuracy is descriptive only and likelihood is the primary outcome, which is weaker evidence of score inflation.
- **Validation is limited:** it measures only the precision of the flag. Recall on clean items is out of scope, and n-gram overlap misses paraphrased contamination.
- **The reference model's** (Qwen2.5-1.5B) training data is undisclosed and may itself be contaminated.
- **Perturbation coverage is partial:** about 34% of GSM8K items. HumanEval fallback windows are shorter (16 tokens), so their overlap is not identical in kind; report the fallback rate.
- **OOM retries** shorten generation for the affected items (logged). **Skipped items** make id sets differ across models, so analyze stops until you rerun with `--skip-existing`.
- **Coverage gaps:** 165/1000 MMLU items have no window. Holm/BH corrections apply within each likelihood output file only.
