@echo off
REM ===========================================================================
REM run_overlap.bat - n-gram overlap for all benchmarks via the infini-gram API,
REM with automatic restarts.
REM
REM Output : overlap.csv, overlap_cache.json (shared cache), overlap_failed.log
REM          (windows that failed), overlap_fallback.json / .log and
REM          overlap_fallback_summary.csv (HumanEval windows replaced by shorter
REM          ones after a long-query rejection), overlap_run.log (console
REM          output of every attempt, appended).
REM
REM RESTART LOOP: overlap.py is rerun until it exits with code 0 (every window
REM          done, overlap.csv written) or 20 restarts have been made, waiting
REM          60 s between attempts. Exit code 3 = windows still missing, so
REM          overlap.csv was not written; each rerun retries them.
REM          (Test overrides: OVERLAP_MAX_RESTARTS, OVERLAP_RESTART_SLEEP.)
REM HOW TO STOP: press Ctrl+C once in this window. overlap.py saves the cache
REM          atomically (.tmp then rename), so it stays valid. cmd then asks
REM          "Terminate batch job (Y/N)?": answer Y, otherwise the loop goes on
REM          with the next attempt. Closing the window also stops everything
REM          (the cache stays valid; at most 199 unsaved entries are lost).
REM          Do NOT kill python from Task Manager unless Ctrl+C fails.
REM RESUME : run this file again. Cached windows are reused (same --k 15).
REM CHECK  : python overlap.py --benchmarks gsm8k mmlu humaneval --corpora pile dolma --status
REM          python check_overlap_health.py        (both read-only)
REM
REM Refuses to start if another overlap.py is running (two writers would
REM overwrite each other's cache).  run_overlap.bat --dry-run  prints the
REM command only.
REM ===========================================================================
setlocal
cd /d "%~dp0"
set "CMD=python -u overlap.py --benchmarks gsm8k mmlu humaneval --corpora pile dolma --limit 1000 --k 15 --workers 2"
set "MAXR=20"
if defined OVERLAP_MAX_RESTARTS set "MAXR=%OVERLAP_MAX_RESTARTS%"
set "WAIT=60"
if defined OVERLAP_RESTART_SLEEP set "WAIT=%OVERLAP_RESTART_SLEEP%"

powershell -NoProfile -Command "$p = Get-CimInstance Win32_Process -Filter \"Name like 'python%%'\" | Where-Object { $_.CommandLine -match 'overlap\.py' }; if ($p) { $p | ForEach-Object { Write-Host ('  running: PID ' + $_.ProcessId + '  ' + $_.CommandLine) }; exit 1 } else { exit 0 }"
if errorlevel 1 (
    echo ERROR: another overlap.py is already running ^(listed above^). Stop it first; never run two at once.
    exit /b 1
)

if /i "%~1"=="--dry-run" (
    echo [dry-run] %CMD% ^>^> overlap_run.log 2^>^&1   ^(restarts: up to %MAXR%, %WAIT% s apart^)
    exit /b 0
)

echo Console output is appended to overlap_run.log ^(watch with: powershell Get-Content overlap_run.log -Wait^)
set /a ATTEMPT=0
:again
set /a ATTEMPT+=1
set /a RESTARTS=ATTEMPT-1
echo [%DATE% %TIME%] attempt %ATTEMPT% ^(restart %RESTARTS% of at most %MAXR%^): %CMD%
>>overlap_run.log echo ===== [%DATE% %TIME%] attempt %ATTEMPT% (restart %RESTARTS% of at most %MAXR%): %CMD%
%CMD% >> overlap_run.log 2>&1
set "RC=%errorlevel%"
>>overlap_run.log echo ===== [%DATE% %TIME%] attempt %ATTEMPT% exited with code %RC%
if "%RC%"=="0" goto :done
if %ATTEMPT% GTR %MAXR% goto :giveup
echo overlap.py exited with code %RC% ^(3 = windows still missing^). Restarting in %WAIT% s; Ctrl+C then Y stops.
powershell -NoProfile -Command "Start-Sleep -Seconds %WAIT%"
goto :again

:done
echo Done after %ATTEMPT% attempt^(s^). Rows written to overlap.csv. Failed windows ^(if any^): overlap_failed.log
endlocal
exit /b 0

:giveup
echo ERROR: overlap.py still failing after %RESTARTS% restarts ^(last exit code %RC%^). See overlap_run.log and
echo overlap_failed.log. To write overlap.csv anyway: python overlap.py ... --allow-partial
endlocal
exit /b 1
