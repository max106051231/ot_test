#!/usr/bin/env bash
# Semi-Shield ISMS — Ollama 後端（Linux / macOS）
set -euo pipefail
cd "$(dirname "$0")"

export LLM_BACKEND="${LLM_BACKEND:-ollama}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:3b}"
export OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT:-300}"
export PORT="${PORT:-2000}"
export LLM_WARMUP="${LLM_WARMUP:-0}"
export LLM_SPEED="${LLM_SPEED:-turbo}"
export OT_ENABLE_RAG="${OT_ENABLE_RAG:-1}"
export ENABLE_GUARDRAIL="${ENABLE_GUARDRAIL:-1}"
export LLM_SYSLOG_MAX_TOKENS="${LLM_SYSLOG_MAX_TOKENS:-960}"
export LLM_MAX_NEW_TOKENS="${LLM_MAX_NEW_TOKENS:-1536}"
export OT_LOG_CTX_CHARS="${OT_LOG_CTX_CHARS:-12000}"
export OT_SYSLOG_CTX_CHARS="${OT_SYSLOG_CTX_CHARS:-12000}"
export OT_WARROOM_CTX_LIMIT="${OT_WARROOM_CTX_LIMIT:-12000}"
export OT_WARROOM_BUNDLE_CHARS="${OT_WARROOM_BUNDLE_CHARS:-1200}"
export GUARDRAIL_DEVICE="${GUARDRAIL_DEVICE:-cuda}"
export HALLUCINATION_DEVICE="${HALLUCINATION_DEVICE:-cuda}"
export RAG_DEVICE="${RAG_DEVICE:-cuda}"
export OLLAMA_NUM_GPU="${OLLAMA_NUM_GPU:--1}"
export OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-4096}"
export OLLAMA_GEMMA4_THINK="${OLLAMA_GEMMA4_THINK:-0}"
export OLLAMA_GEMMA4_THINK_TOKENS="${OLLAMA_GEMMA4_THINK_TOKENS:-1400}"
export OLLAMA_GEMMA4_NUM_CTX="${OLLAMA_GEMMA4_NUM_CTX:-8192}"
export LLM_HOT_SWITCH="${LLM_HOT_SWITCH:-1}"
export LLM_RELEASE_ON_SWITCH="${LLM_RELEASE_ON_SWITCH:-0}"
export LLM_MODEL_CACHE_MAX="${LLM_MODEL_CACHE_MAX:-32}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"
export OLLAMA_PRELOAD_ALL="${OLLAMA_PRELOAD_ALL:-0}"
export ENABLE_AI_REVIEWER="${ENABLE_AI_REVIEWER:-1}"
export KILL_STALE_PORT="${KILL_STALE_PORT:-1}"

PY="$(python3 scripts/platform_utils.py python-path 2>/dev/null || command -v python3)"
echo "[Python] $($PY --version)"
echo "[Ollama] backend=$LLM_BACKEND model=$OLLAMA_MODEL url=$OLLAMA_BASE_URL"
echo "Make sure Ollama is running: ollama serve  (or Ollama desktop)"

"$PY" scripts/platform_utils.py kill-port "$PORT" || true
exec "$PY" app.py
