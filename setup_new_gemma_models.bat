@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY=py -3.12"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

echo Gemma 3/4 setup: download, fine-tune, register Ollama
echo Requires: ollama serve, HF_TOKEN for google/* gated models
echo.

"%PY%" scripts\setup_new_gemma_models.py %*
pause
