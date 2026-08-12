@echo off
setlocal EnableExtensions
REM Semi-Shield ISMS - Ollama backend (ASCII-only for cmd.exe)
cd /d "%~dp0"

if "%LLM_BACKEND%"=="" set "LLM_BACKEND=ollama"
if "%OLLAMA_BASE_URL%"=="" set "OLLAMA_BASE_URL=http://127.0.0.1:11434"
if "%OLLAMA_MODEL%"=="" set "OLLAMA_MODEL=qwen2.5:3b"
if "%OLLAMA_TIMEOUT%"=="" set "OLLAMA_TIMEOUT=300"
if "%PORT%"=="" set "PORT=2000"
if "%LLM_WARMUP%"=="" set "LLM_WARMUP=0"
if "%LLM_SPEED%"=="" set "LLM_SPEED=turbo"
if "%OT_ENABLE_RAG%"=="" set "OT_ENABLE_RAG=1"
if "%ENABLE_GUARDRAIL%"=="" set "ENABLE_GUARDRAIL=1"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  set "PY=py -3.12"
)

echo [Python]
"%PY%" --version
echo [Ollama] backend=%LLM_BACKEND% model=%OLLAMA_MODEL% url=%OLLAMA_BASE_URL%
echo Make sure Ollama is running (ollama serve or Ollama desktop)
"%PY%" app.py
pause
exit /b 0
