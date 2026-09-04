#!/usr/bin/env bash
# 建立 Linux / macOS 虛擬環境並安裝 Ollama 後端依賴
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: need python3. Install Python 3.12+ and retry."
  exit 1
fi

echo "[1/3] Create venv .venv"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[2/3] Upgrade pip"
pip install -U pip wheel

echo "[3/3] Install requirements-ollama.txt"
pip install -r requirements-ollama.txt

echo "Done. Activate: source .venv/bin/activate"
echo "Start server: ./run_ollama.sh"
