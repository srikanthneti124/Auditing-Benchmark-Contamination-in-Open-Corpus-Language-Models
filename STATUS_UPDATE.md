> **SUPERSEDED (2026-10-29).** Historical record of an earlier state; commands, defaults and file formats below may no longer be current. Current behaviour: README.md and PREFLIGHT.md; current status: STATUS_FINAL.md.

# STATUS_UPDATE — coding phases (2026-10-25)

All claims below come from command output in this session. PID 28348 (old `overlap.py`) was **not** touched and is still running: the cache file's mtime was 02:58:06, which I read from file metadata only.

**One rule breach to disclose:** an early version of a new test called `monkeypatch.undo()`, which reset the working directory. Its final assertion then **read** the live `overlap_cache.json` once (a read only; the only write happened earlier, in the temp folder). That test now uses a scoped patch, plus a guard that fails any test if the working directory is the project folder.

## 1. Phase results
| Phase | Result | Evidence |
|---|---|---|
| 1 Parallel overlap.py | **PASS** | The file already had parallel queries from my earlier session; edited in place: `--workers` default 2, optional `truststore`, explicit retry on 403/429/5xx. `tests/test_overlap_parallel.py`: **14 passed**, covering (a) rows identical to the old sequential loop, (b) cache never shrinks, (c) cache parses after a simulated Ctrl+C and after an interrupted save, (d) failures logged and excluded from `n_windows`, not counted as 0, (e) `--k 1` and HumanEval code windows. |
| 2 Resumable scripts and validation | **PASS** | `--skip-existing` in `evaluate.py` and `itemloglik.py`: rerun → "nothing new"; a deleted row is restored once; the default still appends. `analyze.py` exits on duplicate keys, differing id sets, or a missing corpus, and prints/saves `strata_n.csv` with an n<30 flag. 13 new tests. |
| 3 GPU | **PASS** | `True NVIDIA GeForce RTX 5070 Laptop GPU 13.0`, torch `2.14.1+cu130`, from the official `whl/cu130` index (the driver reports CUDA 13.1). 4096² fp32 matmul: 10.0 ms on GPU vs 197.6 ms on CPU. Forward pass: 667 MiB allocated, logits on `cuda:0`. itemloglik `--limit 20`: **6.3 s GPU (fp16) vs 10.8 s CPU**. `--device`, `--batch-size` and halving on OOM are implemented; the OOM path is **UNVERIFIED** (never triggered). |
| 4 Model plan | **PASS (with open decisions)** | Table below; `models.json`; `--model-corpus-map` with fallback, 3 tests; `report.py` writes `tab_outcome_support.tex` (outcome per model) plus a likelihood figure and table. |
| 5 Local test and offline chain | **PASS** | `pytest`: **506 passed**. `_work/smoke/run_smoke.sh`: every step exit 0, total 4 min 2 s. Six figures and seven tables generated and inspected; fixed accuracy axes going below 0. |
| 6 Handoff | **PASS** | `run_overlap.bat`: the guard refused to start because PID 28348 is running (correct); the dry-run branch, tested on a copy, printed the exact command. `run_after_overlap.bat --dry-run` listed all 17 steps; a copy with a broken step stopped at step 1 with exit 1. `check_overlap_health.py` was tested on a copy of the backup: per-index counts, rate 29.9/min after simulated growth, 27,824 of 48,424 planned windows missing. |

## 2. Changed (a `.bak` of each is in the folder) and new files
| File | Change |
|---|---|
| overlap.py | workers=2, optional truststore, 403/429/5xx retry (on top of the earlier parallel and logging version) |
| evaluate.py | `--skip-existing`, `--device`, `--dtype` (default `auto` = bf16 on GPU, unchanged) |
| itemloglik.py | `--skip-existing`, `--device`, `--batch-size`, fp16 on GPU by default, `--dtype`, OOM halving |
| analyze.py | input validation, n-per-stratum table, `--model-corpus-map`; statistics unchanged |
| report.py | likelihood figure and table, outcome-support table, accuracy axes start at 0 |
| validate_sample.py | `--no-snippets` no longer creates or rewrites `overlap_cache.json` (it previously did, in the working directory) |
| requirements.txt | CUDA install note, optional truststore |
| new | models.json, run_overlap.bat, run_after_overlap.bat, check_overlap_health.py, PREFLIGHT.md, tests/test_overlap_parallel.py, test_analyze_validation.py, test_resume_and_batching.py |

## 3. Tests
**506 passed, 0 failed.** One new test of mine needed a looser tolerance after measurement. On CPU fp32, an unpadded batch of 5 different texts differs from single-text scoring by up to 2.6e-3 nats, so the cause is batch-shape rounding, not padding leakage; batch size 1 is exact. Its tolerance is now `abs=5e-3`, and it also checks exact equality at batch size 1. No pre-existing test was changed.

## 4. Model plan (8 GB). Sources: Hugging Face model cards read this session
| Model | Params (card) | fp16 GB | Fits 8 GB | Corpus (card) | Index |
|---|---|---|---|---|---|
| pythia-410m | 405M | 0.8 | yes | Pile, not deduplicated | v4_piletrain_llama |
| pythia-1b | 1.01B | 2.0 | yes | Pile | same |
| pythia-1.4b | 1.41B | 2.8 | yes (fp32 5.7 GB) | Pile | same |
| pythia-2.8b | 2.78B | 5.6 | fp16 only; UNVERIFIED in practice | Pile | same |
| OLMo-1B-hf | about 1B | about 2.4 | yes | "Dolma", **version not stated** | match **UNVERIFIED**; excluded |
| **OLMo-1B-0724-hf** | about 1B | about 2.4 | yes | "uses the new 1.7 version" | v4_dolma-v1_7_llama, by name; whether the index covers the full training mix is UNVERIFIED |
| Qwen2.5-1.5B (reference) | 1.54B | 3.1 | yes | **not disclosed** | "unrelated corpus" is UNVERIFIED |

Near chance (expected, **UNVERIFIED** until the real runs): every candidate except possibly Qwen on GSM8K/HumanEval, and likely all Pythia models on MMLU. `report.py` labels each model×benchmark: if the CI upper bound of accuracy is ≤ chance + 0.05, the primary outcome becomes the likelihood analysis (log-likelihood and Min-K%).

## 5. Commands you run (details in PREFLIGHT.md)
| # | Command | Time | Hardware |
|---|---|---|---|
| 1 | Ctrl+C on PID 28348, then PREFLIGHT steps 0–5 | about 10 min | CPU, network |
| 2 | `run_overlap.bat` | UNVERIFIED: 27,824 windows remaining; about 10 h at the old job's ~45/min, less with 2 workers if the API allows | network |
| 3 | `python check_overlap_health.py` (repeat) | seconds | CPU |
| 4 | `run_after_overlap.bat --dry-run`, then `run_after_overlap.bat` | UNVERIFIED. Smoke run: pythia-70m took 115 s for 20 items × 3 benchmarks (dominated by GSM8K generation); extrapolation to 1000 items × 6 models is many hours | **GPU** |
| 5 | `validate_sample.py export --n 200` (snippets need the API), annotate, then `score` | about 2 h export plus manual work | network |

## 6. UNVERIFIED
- **API access from your terminal:** PREFLIGHT step 2 verifies it.
- **Real-network behaviour of the parallel `overlap.py`:** PREFLIGHT step 5 verifies it.
- **OOM-halving path:** never triggered.
- **Whether pythia-2.8b and Qwen in fp32 fit in 8 GB during generation:** the first real run shows it.
- **Model run times at full size.**
- **LaTeX tables compile:** `pdflatex` is not installed here.
- **search_docs snippet export:** needs the API.

## 7. Open decisions for you
1. **Approve the model list in `models.json`:** OLMo-1B-0724-hf instead of OLMo-1B-hf, and Qwen2.5-1.5B as a reference whose data is undisclosed.
2. **Precision.** bf16 changed pythia-70m mean log-lik by up to **2.59 nats** and pythia-160m by **0.85**; fp16 by 0.09 and 0.05; GPU fp32 matches CPU to 0.0002. `evaluate.py`'s default bf16 is therefore unsafe for Pythia. The handoff uses `--dtype float32`, with float16 only for 2.8b. Also decide whether to quantize or drop 2.8b.
3. **Near-chance models:** accept likelihood as the primary outcome for them, and decide how the paper presents accuracy where it is uninformative.

## 8. Limitations affecting the paper's claims
- **Pile×GSM8K is almost all clean** (4 of 14,522 Pile windows nonzero), so the Pythia accuracy contrast may be impossible.
- **GSM8K perturbation covers about 34% of items** (selection bias).
- **The reference model's data is unknown**, and it is likely contaminated.
- **Many tests without multiple-comparison correction.**
- **Annotation is unblinded.**
- **165 of 1,000 MMLU items have no windows.**
- **Overlap on the old job's path loses windows silently:** its failures were never logged, so `n_windows` is reduced without a record, as the earlier 105 single-window gaps suggest.
