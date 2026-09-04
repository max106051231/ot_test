@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY=py -3.12"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

set "PYTHONIOENCODING=utf-8"
echo Merge LoRA adapters into train_ai\models ...
"%PY%" scripts\merge_all_to_train_ai_models.py %*
pause
