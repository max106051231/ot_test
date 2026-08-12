@echo off
setlocal EnableExtensions
REM Import Semi-Shield local models into Ollama (ASCII-only for cmd.exe)
cd /d "%~dp0"

echo [1/2] Checking Ollama...

set "OLLAMA_EXE="
where ollama >nul 2>&1
if not errorlevel 1 (
  for /f "delims=" %%O in ('where ollama 2^>nul') do (
    set "OLLAMA_EXE=%%O"
    goto :ollama_found
  )
)
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
  set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
  set "PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%"
  goto :ollama_found
)
if exist "C:\Program Files\Ollama\ollama.exe" (
  set "OLLAMA_EXE=C:\Program Files\Ollama\ollama.exe"
  set "PATH=C:\Program Files\Ollama;%PATH%"
  goto :ollama_found
)
echo ERROR: ollama not found. Install from https://ollama.com/download
pause
exit /b 1

:ollama_found
"%OLLAMA_EXE%" list >nul 2>&1
if errorlevel 1 (
  echo ERROR: Cannot reach Ollama. Start Ollama desktop or run: ollama serve
  pause
  exit /b 1
)

echo [2/2] Importing models...
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  set "PY=py -3.12"
)

"%PY%" scripts\import_models_to_ollama.py %*
if errorlevel 1 (
  echo.
  echo Some models failed. Retry one model, for example:
  echo   "%PY%" scripts\import_models_to_ollama.py --only gemma_2b_ot
  pause
  exit /b 1
)

echo.
echo Done. Check installed models: ollama list
pause
exit /b 0
