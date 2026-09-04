#!/usr/bin/env python3
"""
微調 Gemma3/4 OT 模型並合併到 train_ai/models/<slug>/。

目標 slug：
  gemma3_1b_ot, gemma3_4b_ot, gemma4_e2b_ot, gemma4_e4b_ot

用法：
  python scripts/train_gemma_ot_to_models.py --list
  python scripts/train_gemma_ot_to_models.py --only gemma4_e2b_ot
  python scripts/train_gemma_ot_to_models.py
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

GEMMA_OT_JOBS = [
    ("gemma3_1b_ot", "gemma3_1b"),
    ("gemma3_4b_ot", "gemma3_4b"),
    ("gemma4_e2b_ot", "gemma4_e2b"),
    ("gemma4_e4b_ot", "gemma4_e4b"),
]


def _merged(slug: str) -> bool:
    meta = MODELS_DIR / slug / "train_meta.json"
    if not meta.is_file():
        return False
    try:
        return bool(json.loads(meta.read_text(encoding="utf-8")).get("merge_mode"))
    except Exception:
        return False


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    jobs = [(s, b) for s, b in GEMMA_OT_JOBS if not only or s in only]

    if args.list:
        print(f"輸出：{MODELS_DIR}\n")
        for slug, base in jobs:
            merged = _merged(slug)
            base_ok = (MODELS_DIR / base / "config.json").is_file() or (
                MODELS_DIR / slug / "config.json"
            ).is_file()
            print(
                f"  {slug:18s} base={base:12s} "
                f"base_ok={'Y' if base_ok else 'N'} merged={'Y' if merged else 'N'}"
            )
        return 0

    if not args.skip_download:
        bases = sorted({b for _, b in jobs})
        rc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "download_models_to_train_ai.py"),
                "--only",
                ",".join(bases),
            ],
            cwd=str(ROOT),
        ).returncode
        if rc != 0:
            print("warn: some base downloads failed (check HF login)")

    slugs = [s for s, _ in jobs]
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_pending_llm.py"),
        "--only",
        ",".join(slugs),
    ]
    if args.force:
        cmd.append("--force")
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode

    print(f"\nModels directory: {MODELS_DIR}")
    for slug, _ in jobs:
        p = MODELS_DIR / slug
        if _merged(slug):
            print(f"  OK {slug} -> {p}")
        else:
            print(f"  -- {slug} (not merged yet)")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
