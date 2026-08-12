@echo off
REM 建立 Python 3.12 虛擬環境並安裝 Semi-Shield 相依套件
cd /d "%~dp0"

echo === Semi-Shield Python 3.12 環境 ===
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo [錯誤] 找不到 Python 3.12。請從 Microsoft Store 或 python.org 安裝 3.12。
    pause
    exit /b 1
)

echo [1/4] 建立 .venv（Python 3.12）...
py -3.12 -m venv .venv
if errorlevel 1 (
    echo [錯誤] venv 建立失敗
    pause
    exit /b 1
)

set "PIP=.venv\Scripts\pip.exe"
set "PY=.venv\Scripts\python.exe"

echo [2/4] 升級 pip...
"%PY%" -m pip install -U pip wheel setuptools

echo [3/4] 安裝 PyTorch（CPU 版，供護欄/RAG）...
"%PIP%" install torch --index-url https://download.pytorch.org/whl/cpu

echo [4/4] 安裝平台相依（Ollama 模式）...
"%PIP%" install -r requirements-ollama.txt

echo.
echo === 完成 ===
"%PY%" --version
"%PY%" -c "import numpy, torch; print('numpy', numpy.__version__, 'torch', torch.__version__)"
echo.
echo 之後請用 run_ollama.bat 啟動（會自動使用 .venv）。
pause
