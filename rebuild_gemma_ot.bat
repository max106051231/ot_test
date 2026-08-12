@echo off
setlocal EnableExtensions
REM Rebuild gemma_2b_ot as gemma2:2b + Semi-Shield SYSTEM (fixes broken safetensors import)
cd /d "%~dp0"

set "OLLAMA_EXE="
where ollama >nul 2>&1
if not errorlevel 1 (
  for /f "delims=" %%O in ('where ollama 2^>nul') do set "OLLAMA_EXE=%%O" & goto :found
)
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
  set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
  goto :found
)
echo ERROR: ollama not found
pause
exit /b 1

:found
echo Removing broken gemma_2b_ot safetensors model (if any)...
"%OLLAMA_EXE%" rm gemma_2b_ot >nul 2>&1

echo Creating gemma_2b_ot from gemma2:2b + Semi-Shield SYSTEM...
"%OLLAMA_EXE%" create gemma_2b_ot -f scripts\ollama_modelfiles\gemma_2b_ot.Modelfile
if errorlevel 1 (
  echo Failed. Try: ollama pull gemma2:2b
  pause
  exit /b 1
)

echo.
echo Done. gemma_2b_ot is now a wrapper around gemma2:2b.
echo Note: true LoRA weights stay in train_ai\train_llm\outputs\gemma_2b_ot
echo       Ollama cannot run that LoRA directly; use Qwen OT models for full fine-tune via Ollama.
echo.
"%OLLAMA_EXE%" list | findstr /i gemma
pause
exit /b 0
