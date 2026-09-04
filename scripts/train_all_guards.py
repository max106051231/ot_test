#!/usr/bin/env python3
"""一鍵訓練輸入護欄 + 輸出幻覺偵測模型。"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], cwd: Path) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(cwd))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-input-guard",
        action="store_true",
        help="略過輸入護欄（已有 fine_tuned_guardrail 時）",
    )
    parser.add_argument("--guard-epochs", type=int, default=3)
    parser.add_argument("--hallucination-epochs", type=int, default=4)
    args = parser.parse_args()

    py = sys.executable
    gur = ROOT / "train_ai" / "train_gur"
    hall = ROOT / "train_ai" / "train_hallucination"
    guard_weights = gur / "fine_tuned_guardrail" / "model.safetensors"

    if not args.skip_input_guard and not guard_weights.is_file():
        print("=== 1/2 輸入護欄（safe/unsafe）===")
        if not (gur / "guardrail_dataset.json").exists():
            run([py, "generate_dataset.py"], gur)
        run(
            [py, "train_Guard.py", "--model", "hfl/chinese-roberta-wwm-ext"],
            gur,
        )
    else:
        print("=== 1/2 輸入護欄：略過（已有權重或 --skip-input-guard）===")

    print("\n=== 2/2 輸出幻覺偵測（grounded/hallucinated）===")
    run([py, "generate_hallucination_dataset.py"], hall)
    run(
        [py, "train_hallucination.py", "--epochs", str(args.hallucination_epochs)],
        hall,
    )

    print("\n完成。請重啟 app.py（ENABLE_GUARDRAIL=1 ENABLE_HALLUCINATION_GUARD=1）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
