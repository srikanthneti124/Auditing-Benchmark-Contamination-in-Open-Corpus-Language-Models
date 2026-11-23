@echo off
REM ===========================================================================
REM run_after_overlap.bat - kept for compatibility. Runs run_gpu_stages.bat
REM (perturb, evaluate, itemloglik; does not need overlap.csv), then
REM run_analysis.bat (analyze, report; needs overlap.csv). See those files.
REM The previous single-file version is run_after_overlap.bat.bak.
REM
REM Usage:  run_after_overlap.bat            run everything
REM         run_after_overlap.bat --dry-run  print the commands only
REM ===========================================================================
setlocal
cd /d "%~dp0"
call "%~dp0run_gpu_stages.bat" %1
if errorlevel 1 exit /b 1
call "%~dp0run_analysis.bat" %1
if errorlevel 1 exit /b 1
exit /b 0
