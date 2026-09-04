#!/usr/bin/env python3
"""
以 train_ai/train_llm/train.jsonl 微調「尚未 LoRA 訓練過」的地端模型。

判定：train_meta.json 含 merge_mode=full_precision_base_plus_lora 或
      outputs/<slug>/lora_adapter 存在 → 視為已微調，預設跳過。

用法（專案根目錄）：
  python scripts/train_pending_llm.py --list
  python scripts/train_pending_llm.py
  python scripts/train_pending_llm.py --only llama32_3b_ot,gemma2_9b_ot
  python scripts/train_pending_llm.py --force          # 含已微調者重訓
  python scripts/train_pending_llm.py --max-steps 150 --batch-size 1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = ROOT / "train_ai" / "train_llm" / "train.py"
MODELS_DIR = ROOT / "train_ai" / "models"
OUTPUTS_DIR = ROOT / "train_ai" / "train_llm" / "outputs"
DATA_PATH = ROOT / "train_ai" / "train_llm" / "train.jsonl"

# slug, hf_model_id, max_steps, batch_size, max_length
PENDING_JOBS: list[dict] = [
    {
        "slug": "phi4_mini_ot",
        "model_id": "microsoft/Phi-4-mini-instruct",
        "local_base": "phi4_mini_ot",
        "max_steps": 200,
        "batch_size": 2,
        "max_length": 2048,
    },
    {
        "slug": "llama32_3b_ot",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "local_base": "llama32_3b_ot",
        "max_steps": 200,
        "batch_size": 2,
        "max_length": 2048,
    },
    {
        "slug": "llama32_1b_ot",
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "local_base": "llama32_1b",
        "max_steps": 200,
        "batch_size": 2,
        "max_length": 2048,
    },
    {
        "slug": "gemma2_2b_ot",
        "model_id": "google/gemma-2-2b-it",
        "local_base": "gemma2_2b",
        "max_steps": 200,
        "batch_size": 2,
        "max_length": 2048,
    },
    {
        "slug": "gemma2_9b_ot",
        "model_id": "google/gemma-2-9b-it",
        "local_base": "gemma2_9b",
        "max_steps": 150,
        "batch_size": 1,
        "max_length": 1536,
    },
    {
        "slug": "gemma3_1b_ot",
        "model_id": "google/gemma-3-1b-it",
        "local_base": "gemma3_1b",
        "max_steps": 200,
        "batch_size": 2,
        "max_length": 2048,
    },
    {
        "slug": "gemma3_4b_ot",
        "model_id": "google/gemma-3-4b-it",
        "local_base": "gemma3_4b",
        "max_steps": 200,
        "batch_size": 1,
        "max_length": 2048,
    },
    {
        "slug": "gemma4_e2b_ot",
        "model_id": "google/gemma-4-E2B-it-qat-q4_0-unquantized",
        "local_base": "gemma4_e2b",
        "max_steps": 150,
        "batch_size": 1,
        "max_length": 1536,
    },
    {
        "slug": "gemma4_e4b_ot",
        "model_id": "google/gemma-4-E4B-it-qat-q4_0-unquantized",
        "local_base": "gemma4_e4b",
        "max_steps": 120,
        "batch_size": 1,
        "max_length": 1536,
    },
]


def _is_finetuned(slug: str) -> bool:
    adapter = OUTPUTS_DIR / slug / "lora_adapter" / "adapter_config.json"
    if adapter.is_file():
        return True
    meta_path = MODELS_DIR / slug / "train_meta.json"
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if meta.get("source") == "huggingface_download":
        return False
    return bool(meta.get("merge_mode") or meta.get("adapter_dir"))


def _has_local_base(local_slug: str) -> bool:
    p = MODELS_DIR / local_slug
    if not p.is_dir() or not (p / "config.json").is_file():
        return False
    return bool(
        (p / "model.safetensors").is_file()
        or list(p.glob("model-*.safetensors"))
        or list(p.glob("pytorch_model*.bin"))
    )


def list_jobs(force: bool) -> None:
    print(f"資料集：{DATA_PATH} （存在={DATA_PATH.is_file()}）\n")
    print("待微調模型：")
    for job in PENDING_JOBS:
        slug = job["slug"]
        fin = _is_finetuned(slug)
        base_ok = _has_local_base(job.get("local_base") or slug)
        status = "已微調（跳過）" if fin and not force else "待訓練"
        if not base_ok and not fin:
            status = "缺本地基底（將從 HF 下載）"
        print(
            f"  - {slug:16s}  base={job['model_id']}\n"
            f"    local={job.get('local_base')}  steps={job['max_steps']}  "
            f"batch={job['batch_size']}  → {status}"
        )


def run_job(job: dict, *, data: Path, force: bool, skip_merge: bool) -> int:
    slug = job["slug"]
    if _is_finetuned(slug) and not force:
        print(f"⏭️  跳過 {slug}（已有 LoRA／merge 紀錄；加 --force 可重訓）")
        return 0

    cmd = [
        sys.executable,
        str(TRAIN_PY),
        "--model",
        job["model_id"],
        "--slug",
        slug,
        "--data",
        str(data),
        "--max-steps",
        str(job["max_steps"]),
        "--batch-size",
        str(job["batch_size"]),
        "--max-length",
        str(job.get("max_length") or 2048),
    ]
    if skip_merge:
        cmd.append("--skip-merge")

    print("\n" + "=" * 72)
    print(f"▶ 開始微調：{slug}")
    print("=" * 72)
    proc = subprocess.run(cmd, cwd=str(TRAIN_PY.parent))
    return int(proc.returncode or 0)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="微調尚未 LoRA 訓練的地端模型")
    parser.add_argument("--list", action="store_true", help="列出狀態")
    parser.add_argument("--only", type=str, default="", help="逗號分隔 slug")
    parser.add_argument("--force", action="store_true", help="已微調也重訓")
    parser.add_argument("--skip-merge", action="store_true", help="只存 adapter")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    if not DATA_PATH.is_file():
        print(f"❌ 找不到 {DATA_PATH}")
        return 1

    if args.list:
        list_jobs(args.force)
        return 0

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    jobs = [j for j in PENDING_JOBS if not only or j["slug"] in only]
    if not jobs:
        print("❌ 無符合的訓練工作（--only / PENDING_JOBS）")
        return 1

    if args.max_steps is not None:
        for j in jobs:
            j["max_steps"] = args.max_steps
    if args.batch_size is not None:
        for j in jobs:
            j["batch_size"] = args.batch_size

    print(f"將訓練 {len(jobs)} 個模型：{[j['slug'] for j in jobs]}")
    failed: list[str] = []
    for job in jobs:
        rc = run_job(
            job,
            data=DATA_PATH,
            force=args.force,
            skip_merge=bool(args.skip_merge),
        )
        if rc != 0:
            failed.append(job["slug"])
            print(f"❌ {job['slug']} 失敗（exit={rc}）")

    if failed:
        print(f"\n完成（失敗：{', '.join(failed)}）")
        return 1
    print("\n✅ 全部微調完成。請執行：python scripts/import_models_to_ollama.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
