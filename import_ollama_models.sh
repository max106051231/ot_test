#!/usr/bin/env bash
# 將 Semi-Shield 微調模型匯入 Ollama（Linux / macOS）
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: ollama not found. Install from https://ollama.com/download"
  exit 1
fi

if ! ollama list >/dev/null 2>&1; then
  echo "ERROR: Cannot reach Ollama. Start: ollama serve"
  exit 1
fi

PY="$(python3 scripts/platform_utils.py python-path 2>/dev/null || command -v python3)"
echo "[Import] using $PY"
"$PY" scripts/import_models_to_ollama.py "$@"
echo "Done. Check: ollama list"
