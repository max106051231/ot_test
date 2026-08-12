"""
Qwen3-4B-Instruct-2507 LoRA 微調（推薦 #1）

適合：繁中 OT／ISO 27001 指令微調、單卡 16GB（RTX 5070 Ti）QLoRA。
基底：Qwen/Qwen3-4B-Instruct-2507（Apache-2.0，中文與長上下文佳）

用法：
  cd train_ai/train_llm
  python finetune_qwen3_4b.py
  python finetune_qwen3_4b.py --max-steps 300 --batch-size 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

from train import DEFAULT_DATA, MODELS_DIR, resolve_data_path, train_one

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SLUG = "qwen3_4b_ot"
DEFAULT_MAX_STEPS = 250
DEFAULT_BATCH_SIZE = 2
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

    print(f"[推薦模型 #1] {MODEL_ID}")
    print("理由：Qwen3 指令版、繁中強、授權友善、16GB 顯存可穩定 QLoRA。")
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
