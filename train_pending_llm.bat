@echo off
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=py -3.12"
echo 使用 train.jsonl 微調尚未 LoRA 的地端模型...
"%PY%" scripts\train_pending_llm.py %*
pause
