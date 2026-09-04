#!/usr/bin/env python3
"""
下載 Gemma 3/4 基底 → LoRA 微調 → 註冊 Ollama。

類似 gemma4:e4b-it-qat 的新世代模型：
  - gemma3_1b_ot  ← google/gemma-3-1b-it
  - gemma3_4b_ot  ← google/gemma-3-4b-it
  - gemma4_e2b_ot ← google/gemma-4-E2B-it-qat-q4_0-unquantized
  - gemma4_e4b_ot ← google/gemma-4-E4B-it-qat-q4_0-unquantized

用法（專案根目錄）：
  python scripts/setup_new_gemma_models.py --list
  python scripts/setup_new_gemma_models.py --register-only
  python scripts/setup_new_gemma_models.py --only gemma3_1b_ot,gemma4_e4b_ot
  python scripts/setup_new_gemma_models.py --skip-train
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SLUG_TO_LOCAL = {
    "gemma3_1b_ot": "gemma3_1b",
    "gemma3_4b_ot": "gemma3_4b",
    "gemma4_e2b_ot": "gemma4_e2b",
    "gemma4_e4b_ot": "gemma4_e4b",
}

NEW_SLUGS = list(SLUG_TO_LOCAL.keys())

BASE_ALIASES = [
    "base_gemma3_1b_qat",
    "base_gemma3_4b_qat",
    "base_gemma4_e2b_qat",
    "base_gemma4_e4b_qat",
]


def _run(cmd: list[str], *, check: bool = True) -> int:
    print(f"\n>> {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if check and proc.returncode:
        raise SystemExit(proc.returncode)
    return int(proc.returncode or 0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Gemma 3/4 微調並註冊 Ollama")
    ap.add_argument("--list", action="store_true", help="列出待處理模型")
    ap.add_argument("--only", type=str, default="", help="逗號分隔 slug")
    ap.add_argument("--register-only", action="store_true", help="只 pull Ollama 並建立 OT 別名")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-import", action="store_true")
    ap.add_argument("--force", action="store_true", help="已微調也重訓")
    args = ap.parse_args()

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    slugs = [s for s in NEW_SLUGS if not only or s in only]
    if not slugs:
        print("❌ 無符合的 slug")
        return 1

    if args.list:
        _run([sys.executable, "scripts/train_pending_llm.py", "--list"], check=False)
        print("\n新世代 Gemma OT slug：", ", ".join(slugs))
        return 0

    if not args.register_only and not args.skip_download:
        locals_wanted = sorted({SLUG_TO_LOCAL[s] for s in slugs if s in SLUG_TO_LOCAL})
        if locals_wanted:
            _run([
                sys.executable,
                "scripts/download_models_to_train_ai.py",
                "--only",
                ",".join(locals_wanted),
            ])

    if not args.register_only and not args.skip_train:
        cmd = [
            sys.executable,
            "scripts/train_pending_llm.py",
            "--only",
            ",".join(slugs),
        ]
        if args.force:
            cmd.append("--force")
        rc = _run(cmd, check=False)
        if rc != 0:
            print("⚠️  部分微調失敗；仍會嘗試 Ollama 註冊（fallback 至 QAT 基底別名）")

    if not args.skip_import:
        import_targets = slugs + BASE_ALIASES
        _run([
            sys.executable,
            "scripts/import_models_to_ollama.py",
            "--only",
            ",".join(import_targets),
        ], check=False)

    print("\nDone. Run: ollama list  |  run_ollama.bat  |  switch model in Agent menu.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
