@echo off
REM 專案統一 Python：優先 .venv（3.12），其次 py -3.12
set "ROOT=%~dp0.."
if exist "%ROOT%\.venv\Scripts\python.exe" (
    "%ROOT%\.venv\Scripts\python.exe" %*
) else (
    py -3.12 %*
)
