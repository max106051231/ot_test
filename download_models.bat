@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY=py -3.12"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

echo Downloading models to train_ai\models ...
"%PY%" scripts\download_models_to_train_ai.py --skip-existing %*
if errorlevel 1 (
  echo.
  echo Some models failed. For Llama, accept HF license then: huggingface-cli login
  pause
  exit /b 1
)
echo Done.
pause
