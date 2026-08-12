"""
Qwen2.5-7B-Instruct LoRA 微調（推薦 #2）

適合：想比 3B/4B 再拉高報告品質與專業深度；單卡 16GB 用 QLoRA（batch=1）。
基底：Qwen/Qwen2.5-7B-Instruct（成熟、繁中與工具指令表現穩定）

用法：
  cd train_ai/train_llm
  python finetune_qwen25_7b.py
  python finetune_qwen25_7b.py --max-steps 200 --batch-size 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

from train import DEFAULT_DATA, MODELS_DIR, resolve_data_path, train_one

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SLUG = "qwen25_7b_ot"
DEFAULT_MAX_STEPS = 200
DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_LENGTH = 2048


def parse_args():
    p = argparse.ArgumentParser(
        description=f"Fine-tune {MODEL_ID} with QLoRA for OT / ISO 27001"
    )
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--slug", type=str, default=SLUG)
    p.add_argument("--output-root", type=str, default=str(MODELS_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    data_path = resolve_data_path(args.data)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (Path(__file__).resolve().parent / output_root).resolve()

    print(f"[推薦模型 #2] {MODEL_ID}")
    print("理由：7B 容量更大，合規報告與技術深度通常優於 3B/4B；16GB 請用 batch=1。")
    train_one(
        model_id=MODEL_ID,
        slug=args.slug,
        data_path=data_path,
        output_root=output_root,
        max_steps=args.max_steps,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
