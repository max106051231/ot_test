@echo off
setlocal EnableExtensions
REM Semi-Shield ISMS - Ollama backend (ASCII-only for cmd.exe)
cd /d "%~dp0"

if "%LLM_BACKEND%"=="" set "LLM_BACKEND=ollama"
if "%OLLAMA_BASE_URL%"=="" set "OLLAMA_BASE_URL=http://127.0.0.1:11434"
if "%OLLAMA_MODEL%"=="" set "OLLAMA_MODEL=llama3.2:3b"
if "%OLLAMA_TIMEOUT%"=="" set "OLLAMA_TIMEOUT=300"
if "%PORT%"=="" set "PORT=2000"
if "%LLM_WARMUP%"=="" set "LLM_WARMUP=0"
if "%LLM_SPEED%"=="" set "LLM_SPEED=turbo"
if "%OT_ENABLE_RAG%"=="" set "OT_ENABLE_RAG=1"
if "%ENABLE_GUARDRAIL%"=="" set "ENABLE_GUARDRAIL=1"
if "%LLM_SYSLOG_VERBOSE%"=="" set "LLM_SYSLOG_VERBOSE=1"
if "%LLM_SYSLOG_MAX_TOKENS%"=="" set "LLM_SYSLOG_MAX_TOKENS=960"
if "%LLM_SYSLOG_MAX_CHARS%"=="" set "LLM_SYSLOG_MAX_CHARS=5200"
if "%SYSLOG_CTX_CHARS%"=="" set "SYSLOG_CTX_CHARS=18000"
if "%LLM_MAX_NEW_TOKENS%"=="" set "LLM_MAX_NEW_TOKENS=960"

REM 護欄／RAG／幻覺檢測走 GPU（與 Ollama LLM 同卡）
if "%GUARDRAIL_DEVICE%"=="" set "GUARDRAIL_DEVICE=cuda"
if "%HALLUCINATION_DEVICE%"=="" set "HALLUCINATION_DEVICE=cuda"
if "%RAG_DEVICE%"=="" set "RAG_DEVICE=cuda"
REM Ollama 推論：全層 GPU；Gemma3/4 縮短 ctx 加速
if "%OLLAMA_NUM_GPU%"=="" set "OLLAMA_NUM_GPU=-1"
if "%OLLAMA_NUM_CTX%"=="" set "OLLAMA_NUM_CTX=4096"
if "%OLLAMA_KEEP_ALIVE%"=="" set "OLLAMA_KEEP_ALIVE=30m"

if "%ENABLE_AI_REVIEWER%"=="" set "ENABLE_AI_REVIEWER=1"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  set "PY=py -3.12"
)

REM 若埠號已被舊版 app.py 佔用，先關閉以免 API 缺路由
"%PY%" scripts\platform_utils.py kill-port %PORT% 2>nul

echo [Python]
"%PY%" --version
echo [Ollama] backend=%LLM_BACKEND% model=%OLLAMA_MODEL% url=%OLLAMA_BASE_URL%
echo Make sure Ollama is running (ollama serve or Ollama desktop)
"%PY%" app.py
pause
exit /b 0
