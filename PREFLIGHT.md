# PREFLIGHT: run these in your own terminal, in order

Run from the project folder (`cd C:\Users\jahna\OneDrive\Pictures\Desktop\3`) in **cmd** or **PowerShell**. Each step says what you should see. Stop at the first step that differs. Full run order: README.md.

## 0. Power and disk
- Plugged in; Settings > System > Power: **Sleep = Never** while jobs run.
- OneDrive: pause syncing for the duration of the runs (ONEDRIVE_NOTES.md).
- Model weights need about **19.8 GB** (see README.md, "Storage"). Run `python check_storage.py` to see free space and PASS/FAIL per model (3 GB margin). Free space on C:, or point the Hugging Face cache to a folder you choose, e.g. `setx HF_HOME D:\hf_cache` (then open a new terminal), and run `check_storage.py` again. This is your decision; nothing has been moved.

## 1. No overlap.py running
```
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object { $_.CommandLine -match 'overlap\.py' } | Select ProcessId,CommandLine"
```
**Expect:** no output.

## 2. API reachable from *your* terminal
```
python preflight_api.py
```
It applies truststore if installed, sends a count query of the US Constitution preamble to both indices (Pile and Dolma), and one search_docs call, and prints HTTP status, count and latency for each.
**Expect:** two `[count] ... OK` lines with count > 0, one `[search_docs] ... OK` line, then `PREFLIGHT OK`, exit code 0. Any `FAIL` (for example status=403) means do not start the overlap run.

## 3. The current overlap.py is in place
```
python overlap.py --help | findstr /i "workers allow-partial status"
```
**Expect:** lines for `--workers`, `--allow-partial` and `--status`.

## 4. Unit tests (offline, about 3 min, uses the GPU if present)
```
python -m pytest tests -q
```
**Expect:** `0 failed` (the count at handoff is in STATUS_FINAL.md).

## 5. Where the overlap run stands (read-only, no API)
```
python overlap.py --benchmarks gsm8k mmlu humaneval --corpora pile dolma --status
```
**Expect:** a table of expected / done / failed / remaining windows per benchmark x corpus (about 24,200 remaining on 2026-10-29).

## 6. Start the overlap run (restart loop)
```
run_overlap.bat
```
Reruns overlap.py until it exits with code 0 (every window done, overlap.csv written) or 20 restarts, 60 s apart; output is appended to `overlap_run.log`. Stop: Ctrl+C, then answer **Y** to "Terminate batch job". Monitor with `python check_overlap_health.py` (read-only; run it twice, some minutes apart, for a rate and an ETA).

## 7. GPU stages (may run in parallel with step 6)
```
run_gpu_stages.bat --dry-run
run_gpu_stages.bat
```

## 8. After overlap.csv exists and step 7 finished
```
run_analysis.bat --dry-run
run_analysis.bat
```
