#!/usr/bin/env bash
# 樹莓派／無 GPU 啟動腳本
set -euo pipefail
cd "$(dirname "$0")"

export EDGE_MODE=1
export FORCE_CPU=1
export LLM_DEVICE=cpu
export LLM_SPEED="${LLM_SPEED:-edge}"
# 未指定時自動選本機 HF 快取（常見 Qwen2.5-3B-Instruct）
# 強制指定範例：
# export EDGE_LLM_MODEL=Qwen/Qwen2.5-3B-Instruct
# 連網下載小模型：export ALLOW_HF_DOWNLOAD=1 EDGE_LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
export OT_ENABLE_RAG="${OT_ENABLE_RAG:-0}"
export ENABLE_GUARDRAIL="${ENABLE_GUARDRAIL:-0}"
export LLM_WARMUP="${LLM_WARMUP:-0}"
export PORT="${PORT:-2000}"

# 首次會從 HuggingFace 下載小模型；之後可設 HF_HUB_OFFLINE=1
python3 app.py
