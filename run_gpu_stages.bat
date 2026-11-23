@echo off
REM ===========================================================================
REM run_gpu_stages.bat - the stages that need the GPU, in order:
REM   1. perturb_gsm8k.py (generate, then --validate)
REM   2. perturb_humaneval.py
REM   3. evaluate.py and itemloglik.py for every model in models.json
REM      (primary, sensitivity, and the reference last), each at the dtype
REM      listed there, with --skip-existing (resumable; re-running never
REM      duplicates rows). Rows are named <model>@<dtype>, e.g. @fp32.
REM      Per-model max_new_tokens_gsm8k and batch_size come from models.json;
REM      a model with "fits_8gb": false is skipped with a visible message.
REM      STORAGE: check_storage.py runs first (read-only). A model whose
REM      download does not fit in free disk space minus 3 GB (counting the
REM      models before it) is SKIPPED with a visible message, so no download
REM      is started that cannot finish. Plan file: %TEMP%\run_gpu_stages_storage_plan.txt.
REM      Peak GPU memory per benchmark -> gpu_memory_log.csv; items skipped
REM      after CUDA out-of-memory -> generation_skips.csv.
REM None of these scripts reads the overlap output, so this can run while
REM run_overlap.bat is still running. Next step: run_analysis.bat.
REM Deletes nothing. Stops at the first failing step with its name.
REM
REM Usage:  run_gpu_stages.bat            run everything
REM         run_gpu_stages.bat --dry-run  print the commands only
REM ===========================================================================
setlocal
cd /d "%~dp0"
set "LIMIT=1000"
set "SEED=0"
set "BENCH=gsm8k mmlu humaneval"
set "DRY="
if /i "%~1"=="--dry-run" set "DRY=1"

if not exist models.json (
    echo ERROR: models.json not found.
    exit /b 1
)
python modelcfg.py list-run >nul
if errorlevel 1 (
    echo ERROR: models.json could not be read ^(see the message above^).
    exit /b 1
)
echo ===== storage check ^(check_storage.py, downloads nothing^)
python check_storage.py --plan "%TEMP%\run_gpu_stages_storage_plan.txt"
if errorlevel 1 (
    echo ERROR: check_storage.py failed ^(see above^); not starting any model.
    exit /b 1
)

call :step perturb_gsm8k python -u perturb_gsm8k.py --limit %LIMIT% --seed %SEED% --out perturbed_gsm8k.jsonl || goto :fail
call :step perturb_gsm8k_validate python -u perturb_gsm8k.py --validate --out perturbed_gsm8k.jsonl --n-validate 100 --seed %SEED% || goto :fail
call :step perturb_humaneval python -u perturb_humaneval.py --seed %SEED% --out perturbed_humaneval.jsonl || goto :fail

for /f "usebackq tokens=1-6" %%M in ("%TEMP%\run_gpu_stages_storage_plan.txt") do (
    if /i "%%Q"=="no" (
        echo.
        echo ===== SKIPPED %%M: models.json says "fits_8gb": false ^(measured not to fit in 8 GB^)
    ) else if /i not "%%R"=="PASS" (
        echo.
        echo ===== SKIPPED %%M: not enough disk space for its download ^(check_storage.py: %%R; see the table above^)
    ) else (
        call :model "%%M" "%%N" "%%O" "%%P" || goto :fail
    )
)

echo.
if defined DRY (echo DRY RUN: commands listed above, nothing executed.) else (echo GPU STAGES COMPLETED. Next: run_analysis.bat once overlap.csv exists.)
exit /b 0

:model
set "M=%~1"
set "MD=%~2"
set "GT=%~3"
set "BS=%~4"
call :step "evaluate %M% %MD%" python -u evaluate.py --model %M% --benchmarks %BENCH% --limit %LIMIT% --seed %SEED% --dtype %MD% --gsm8k-max-new-tokens %GT% --out results.csv --skip-existing || exit /b 1
call :step "itemloglik %M% %MD%" python -u itemloglik.py --model %M% --benchmarks %BENCH% --limit %LIMIT% --seed %SEED% --dtype %MD% --batch-size %BS% --out itemloglik.csv --skip-existing || exit /b 1
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
