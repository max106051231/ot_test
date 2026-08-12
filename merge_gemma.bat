@echo off
setlocal EnableExtensions
REM Merge Gemma LoRA (outputs) then import to Ollama - ASCII only for cmd.exe
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  set "PY=py -3.12"
)

echo [1/2] Merge Gemma LoRA: outputs/gemma_2b_ot -^> train_ai/models/gemma_2b_ot
echo       Requires HF_TOKEN if google/gemma-2-2b-it is gated.
"%PY%" train_ai\train_llm\train.py --merge-only --slug gemma_2b_ot
if errorlevel 1 (
  echo.
  echo Merge failed. Set HF_TOKEN then retry, or use CPU:
  echo   set HF_TOKEN=your_token
  echo   "%PY%" train_ai\train_llm\train.py --merge-only --slug gemma_2b_ot --merge-device cpu
  pause
  exit /b 1
)

echo.
echo [2/2] Import gemma_2b_ot to Ollama...
call "%~dp0import_ollama_models.bat" --only gemma_2b_ot --skip-merge-lora
exit /b %ERRORLEVEL%
