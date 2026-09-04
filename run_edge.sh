#!/usr/bin/env bash
# 無 GPU / Edge 模式（Linux / macOS / 樹莓派）
set -euo pipefail
cd "$(dirname "$0")"

export EDGE_MODE=1
export FORCE_CPU=1
export LLM_DEVICE=cpu
export LLM_SPEED="${LLM_SPEED:-cpu_fast}"
export OT_ENABLE_RAG="${OT_ENABLE_RAG:-0}"
export ENABLE_GUARDRAIL="${ENABLE_GUARDRAIL:-0}"
export LLM_WARMUP="${LLM_WARMUP:-0}"
export LLM_CHAT_PRIME="${LLM_CHAT_PRIME:-0}"
export LLM_CPU_FAST_GROUNDED="${LLM_CPU_FAST_GROUNDED:-0}"
export LLM_CPU_QUANT="${LLM_CPU_QUANT:-1}"
export PORT="${PORT:-2000}"
export KILL_STALE_PORT="${KILL_STALE_PORT:-1}"

PY="$(python3 scripts/platform_utils.py python-path 2>/dev/null || command -v python3)"
echo "[Edge] FORCE_CPU=1 LLM_DEVICE=cpu model=${EDGE_LLM_MODEL:-auto} speed=$LLM_SPEED"

"$PY" scripts/platform_utils.py kill-port "$PORT" || true
exec "$PY" app.py
