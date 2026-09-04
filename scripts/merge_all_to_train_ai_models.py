#!/usr/bin/env python3
"""
將 train_ai/train_llm/outputs/<slug>/lora_adapter 全精度 merge 到
train_ai/models/<slug>/（微調完模型的標準位置）。

用法（專案根目錄）：
  python scripts/merge_all_to_train_ai_models.py
  python scripts/merge_all_to_train_ai_models.py --only llama32_3b_ot,gemma2_2b_ot
  python scripts/merge_all_to_train_ai_models.py --list
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "train_ai" / "models"
OUTPUTS_DIR = ROOT / "train_ai" / "train_llm" / "outputs"
TRAIN_PY = ROOT / "train_ai" / "train_llm" / "train.py"


def _has_lora(slug: str) -> bool:
    p = OUTPUTS_DIR / slug / "lora_adapter" / "adapter_config.json"
    return p.is_file()


def _is_merged(slug: str) -> bool:
    meta = MODELS_DIR / slug / "train_meta.json"
    if not meta.is_file():
        return False
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(data.get("merge_mode"))


def _base_model_id(slug: str) -> str:
    for cand in (
        OUTPUTS_DIR / slug / "lora_adapter" / "train_meta.json",
        MODELS_DIR / slug / "train_meta.json",
    ):
        if cand.is_file():
            try:
                mid = json.loads(cand.read_text(encoding="utf-8")).get("base_model_id")
                if mid:
                    return str(mid)
            except Exception:
                pass
    presets = ROOT / "train_ai" / "train_llm" / "model_presets.json"
    if presets.is_file():
        data = json.loads(presets.read_text(encoding="utf-8"))
        for meta in (data.get("presets") or {}).values():
            if isinstance(meta, dict) and meta.get("slug") == slug:
                return str(meta.get("model_id") or "")
    return ""


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="LoRA merge 到 train_ai/models")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--merge-device", type=str, default="auto")
    args = ap.parse_args()

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    slugs = sorted(
        p.name
        for p in OUTPUTS_DIR.iterdir()
        if p.is_dir() and _has_lora(p.name)
    )
    if only:
        slugs = [s for s in slugs if s in only]

    if args.list:
        print(f"輸出目錄：{MODELS_DIR}\n")
        for slug in slugs:
            merged = _is_merged(slug)
            print(
                f"  {slug:18s}  lora=是  merged={'是' if merged else '否'}"
                f"  -> {MODELS_DIR / slug}"
            )
        return 0

    if not slugs:
        print("（無 LoRA adapter 可 merge）")
        return 0

    ok, skip, fail = 0, 0, 0
    for slug in slugs:
        out_dir = MODELS_DIR / slug
        if _is_merged(slug) and not args.force:
            print(f"skip {slug} (already merged -> {out_dir})")
            skip += 1
            continue
        model_id = _base_model_id(slug)
        if not model_id:
            print(f"fail {slug}: missing base_model_id")
            fail += 1
            continue
        cmd = [
            sys.executable,
            str(TRAIN_PY),
            "--merge-only",
            "--slug",
            slug,
            "--model",
            model_id,
            "--output-root",
            str(MODELS_DIR),
            "--merge-device",
            args.merge_device,
        ]
        print(f"\nmerge {slug} -> {out_dir}")
        rc = subprocess.run(cmd, cwd=str(TRAIN_PY.parent)).returncode
        if rc == 0:
            ok += 1
        else:
            fail += 1

    print(f"\nDone: ok={ok} skip={skip} fail={fail}")
    print(f"Models dir: {MODELS_DIR}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
