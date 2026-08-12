@echo off
REM 無 GPU / Edge 模式啟動（Windows CUDA 壞掉時也請用這個）
cd /d "%~dp0"

set EDGE_MODE=1
set FORCE_CPU=1
set LLM_DEVICE=cpu
if "%LLM_SPEED%"=="" set LLM_SPEED=edge
REM 速度關鍵：模型越小越快（0.5B >> 1.5B >> 3B）
REM 有快取可指定，例如：
REM   set EDGE_LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
REM   set EDGE_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
REM 桌機可多開執行緒：set LLM_CPU_THREADS=8
REM 關閉 int8 量化：set LLM_CPU_QUANT=0
REM 若要切換「微調後 4-bit」模型：允許暫時改回 GPU
REM   set LLM_ALLOW_GPU_FALLBACK=1
if "%OT_ENABLE_RAG%"=="" set OT_ENABLE_RAG=0
if "%ENABLE_GUARDRAIL%"=="" set ENABLE_GUARDRAIL=0
if "%LLM_WARMUP%"=="" set LLM_WARMUP=0
if "%LLM_CPU_QUANT%"=="" set LLM_CPU_QUANT=1
if "%PORT%"=="" set PORT=2000

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=py -3.12"
echo [Edge] FORCE_CPU=1 LLM_DEVICE=cpu model=%EDGE_LLM_MODEL% speed=%LLM_SPEED%
"%PY%" app.py
pause
