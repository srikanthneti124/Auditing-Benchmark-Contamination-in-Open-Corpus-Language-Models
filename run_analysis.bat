@echo off
REM ===========================================================================
REM run_analysis.bat - after overlap.csv exists AND run_gpu_stages.bat finished:
REM   1. analyze.py --ref-model <reference> --loglik itemloglik.csv
REM      --model-corpus-map models.json --include-sensitivity
REM      main outputs (strata_accuracy.csv, regression.csv, ...) exclude
REM      role "sensitivity" models; those go to *_sensitivity.csv
REM   2. report.py -> paper_outputs\
REM   3. report.py --suffix _sensitivity -> paper_outputs\sensitivity\
REM      (only if analyze.py wrote sensitivity outputs)
REM Partial data is allowed: models with itemloglik.csv rows but no results.csv
REM rows (and a likelihood-only reference model) get likelihood outputs only;
REM see data_availability.csv.
REM Deletes nothing. Stops at the first failing step with its name.
REM
REM Usage:  run_analysis.bat            run everything
REM         run_analysis.bat --dry-run  print the commands only
REM ===========================================================================
setlocal
cd /d "%~dp0"
set "DRY="
if /i "%~1"=="--dry-run" set "DRY=1"

for %%F in (overlap.csv results.csv itemloglik.csv) do (
    if not exist %%F (
        echo ERROR: %%F not found. Run run_overlap.bat and run_gpu_stages.bat first and let them finish.
        if not defined DRY exit /b 1
    )
)
if not exist models.json (
    echo ERROR: models.json not found.
    exit /b 1
)
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name like 'python%%'\" | Where-Object { $_.CommandLine -match 'overlap\.py' }) { exit 1 } else { exit 0 }"
if errorlevel 1 (
    echo ERROR: overlap.py is still running; overlap.csv is not final yet.
    if not defined DRY exit /b 1
)

for /f "usebackq delims=" %%R in (`python modelcfg.py reference`) do set "REF=%%R"
if not defined REF (
    echo ERROR: models.json has no model with role "reference".
    exit /b 1
)

call :step analyze python -u analyze.py --results results.csv --overlap overlap.csv --ref-model %REF% --loglik itemloglik.csv --model-corpus-map models.json --include-sensitivity || goto :fail
call :step report python -u report.py --in-dir . --out-dir paper_outputs || goto :fail
set "SENS="
if exist strata_accuracy_sensitivity.csv set "SENS=1"
if defined DRY set "SENS=1"
if defined SENS (
    call :step report_sensitivity python -u report.py --in-dir . --suffix _sensitivity --out-dir paper_outputs\sensitivity || goto :fail
) else (
    echo note: no *_sensitivity.csv written ^(no sensitivity-model rows^); sensitivity report skipped.
)
echo.
if defined DRY (
    echo DRY RUN: commands listed above, nothing executed.
) else (
    echo ANALYSIS COMPLETED for the data present. Which model has which data ^(accuracy, raw-perturbed
    echo sensitivity, likelihood, reference-model control^): data_availability.csv and
    echo paper_outputs\tab_data_availability.tex. Models without results.csv rows have likelihood outputs only.
)
exit /b 0

:step
set "NAME=%~1"
shift
set "ARGS="
:collect
if "%~1"=="" goto :go
set "ARGS=%ARGS% %1"
shift
goto :collect
:go
echo.
echo ===== %NAME%
echo ^>%ARGS%
if defined DRY exit /b 0
%ARGS%
if errorlevel 1 (
    echo ERROR: step "%NAME%" failed with exit code %errorlevel%.
    exit /b 1
)
exit /b 0

:fail
echo.
echo STOPPED at the first failing step ^(see the message above^). Nothing was deleted; re-run to resume.
exit /b 1
