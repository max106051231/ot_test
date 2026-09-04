@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY=py -3.12"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"

echo ============================================================
echo Gemma OT models -> train_ai\models
echo   gemma3_1b_ot, gemma3_4b_ot, gemma4_e2b_ot, gemma4_e4b_ot
echo ============================================================
echo.
echo NOTE: gemma4 needs ~25GB free disk for HF download.
echo.

"%PY%" scripts\train_gemma_ot_to_models.py %*
pause
