# STATUS_NOW (2026-10-29 03:22)

## 1. Phases
| Phase | Status | Evidence (this session) |
|---|---|---|
| 1 Re-verify | **PASS** | pytest 597 passed (run started 03:03, before this session's edits landed at 03:04). Cache matches the expected hash and mtime. C: 9.30 GB free, D: 146.55 GB. HF_HOME unset (process, user, machine). Hub cache `C:\Users\jahna\.cache\huggingface\hub`; the HF folder holds 4.15 GB. The project **is inside OneDrive**. Only python process: my own pytest. |
| 2 Storage tooling | **PASS** | `check_storage.py` (Hub metadata, no download): 410m / 1b / 1.4b PASS; 2.8b, OLMo, **Qwen (the reference)** FAIL at 10.72 GB free with a 3 GB margin. `run_gpu_stages.bat` skips the FAIL models with a visible message (dry-run checked). `--cache-dir` in evaluate.py, itemloglik.py and gpumem.py. |
| 3 OneDrive safety | **PASS** (real lock UNVERIFIED) | `atomicio.py`: rename retried 10 × 0.2 s; on failure the .tmp is kept, the warning is logged, and the next run merges it. Applied to all 3 atomic writes (overlap cache, fallback map, snippet cache). Mocked-lock tests pass. ONEDRIVE_NOTES.md written. |
| 4 Smoke with models on disk | **PASS**, with 1 issue reported | `--limit 20`, GPU, 0 OOM skips. Peak MiB / seconds per item: 70m eval 330–636 / 0.01–2.5, likelihood ≤ 1663; 410m eval 1732–2167 / 0.06–4.3, likelihood ≤ 3476. All 5 figures and 7 tables opened, non-empty, labelled. Two display bugs fixed (below). Real results.csv, itemloglik.csv and overlap.csv do not exist. |
| 5 Paper drafts | **PASS** | METHODS_DRAFT.md, LIMITATIONS_DRAFT.md (8 listed + 14 found in code), RESULTS_SKELETON.md, CITATIONS_TO_VERIFY.md (21 entries, all UNVERIFIED). No invented numbers; [TO FILL] markers name the source file. |

**Issues found in Phase 4:**
- **Separation (reported, not changed).** pythia-410m on MMLU has 0/5 correct in the contaminated stratum. The logit did not converge (`ConvergenceWarning`), and `regression.csv` / `tab_regression.tex` show OR 0.00 with CI [0.00, inf] and p 1.000 as if fitted. Statistics are untouched per your rule. **Recommended (needs your OK):** an additive `converged` / separation flag column in analyze.py.
- **Fixed, display only:** the forest plot's log axis was wrecked by that fit (tick labels overlapping). Non-estimable fits are now labelled "not estimable" instead of drawn.
- **Fixed, display only:** the likelihood figure title and table caption said "reference model" in a run without one. They now name the controls actually used.

## 2. Files
- **New:**
  - `check_storage.py`: storage check
  - `atomicio.py`: lock-tolerant saves
  - `ONEDRIVE_NOTES.md`
  - Paper drafts: `METHODS_DRAFT.md`, `LIMITATIONS_DRAFT.md`, `RESULTS_SKELETON.md`, `CITATIONS_TO_VERIFY.md`
  - Tests: `tests/test_storage_onedrive.py` (storage, `--cache-dir`, locks), `tests/test_report_display.py`
  - Smoke script and outputs: `_work/smoke/limit20/`
- **Changed** (backups `*.bak_20261029_03xx`, diffs in `_work/diffs_now/`):
  - `overlap.py`, `validate_sample.py`: retrying saves, .tmp recovery
  - `evaluate.py`, `itemloglik.py`, `gpumem.py`: `--cache-dir`
  - `run_gpu_stages.bat`: storage gate, plan file in %TEMP%
  - `models.json`: `download_gb`
  - `report.py`: the two display fixes
  - `README.md`, `PREFLIGHT.md`: storage and OneDrive notes
- **Existing tests:** environment only, no assertion changed (reasoning and diff shown before the edit).
  - `test_dtype_roles_blinding.py`: dry-run env `HF_HUB_OFFLINE=1`, `CHECK_STORAGE_FREE_GB=1000`.
  - `test_gpu_safety.py`: the same env, `check_storage.py` copied, `download_gb` added to the fixture.

## 3. Tests and cache
**621 passed, 0 failed** (final run). Cache SHA-256 `17CB80FDBA17E0AFB4F583501893EC3A072059EA7A508B2EB71A4A69034B47C4`, mtime 2026-10-25 03:39:42: **same at start (03:03) and end (03:22)**.

## 4. UNVERIFIED
| Claim | What would verify it |
|---|---|
| Lock retry works against a real OneDrive / antivirus lock | `atomic_save_warnings.log` during a real run |
| API reachable; real rejection codes | `python preflight_api.py` |
| 1b / 1.4b / OLMo / Qwen / 2.8b fit in GPU memory; full run times | `run_gpu_stages.bat` → `gpu_memory_log.csv` |
| Free space stays as measured (it read 9.30 GB, then 10.72 GB) | `python check_storage.py` right before step 3 below |
| LaTeX compiles | pdflatex |
| Citation titles and claims | CITATIONS_TO_VERIFY.md, checked by you |

## 5. Only you can do next (in order)
1. **Power:** plugged in, Sleep = Never.
2. **OneDrive:** pause syncing for 24 h (ONEDRIVE_NOTES.md).
3. **Storage:** choose a folder with ≥ 22 GB free (19.8 GB of weights + 3 GB margin), `setx HF_HOME <folder>`, open a new terminal, then `python check_storage.py` should show 6 PASS.
4. `python preflight_api.py`
5. `run_overlap.bat`
6. `run_gpu_stages.bat` (can run in parallel with step 5)
7. `run_analysis.bat` (after overlap.csv exists)
8. Blind annotation: `validate_sample.py export` → annotate → `score`
9. Perturbed-item verification: `check_perturbed_sample.py export-gsm8k` / `export-humaneval` → verdicts → `score`
