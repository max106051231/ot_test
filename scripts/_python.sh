#!/usr/bin/env bash
# 專案統一 Python：優先 .venv，其次 python3
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -x "$ROOT/.venv/bin/python3" ]]; then
  exec "$ROOT/.venv/bin/python3" "$@"
fi
exec python3 "$@"
