"""
Gemma-2 LoRA 微調（改自 Qwen 腳本）

基底 google/gemma-2-2b-it 為 Hugging Face **gated** 模型：
  1) 開啟 https://huggingface.co/google/gemma-2-2b-it 同意條款並等審核通過
  2) 建立 token（Read）：https://huggingface.co/settings/tokens
  3) 設定 token 後再訓練（不必裝 huggingface-cli）：
       set HF_TOKEN=hf_xxx
       python finetune_gema.py
     可選：把 token 寫進本機快取
       python -c "from huggingface_hub import login; login(token='hf_xxx')"

若暫時沒有 Gemma 權限，請改用已快取的 Qwen：
  python finetune_qwen3_4b.py
  python finetune_qwen25_7b.py

用法：
  cd train_ai/train_llm
  set HF_TOKEN=你的token
  python finetune_gema.py
  python finetune_gema.py --max-steps 300 --batch-size 2
  python finetune_gema.py --model google/gemma-2-2b-it
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from train import DEFAULT_DATA, MODELS_DIR, resolve_data_path, train_one

MODEL_ID = "google/gemma-2-2b-it"
SLUG = "gemma_2b_ot"
DEFAULT_MAX_STEPS = 250
DEFAULT_BATCH_SIZE = 2
DEFAULT_MAX_LENGTH = 2048


def parse_args():
    p = argparse.ArgumentParser(
        description=f"Fine-tune Gemma with QLoRA for OT / ISO 27001（需 HF gated 權限）"
    )
    p.add_argument("--model", type=str, default=MODEL_ID, help="HuggingFace model id")
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--slug", type=str, default=SLUG)
    p.add_argument("--output-root", type=str, default=str(MODELS_DIR))
    p.add_argument(
        "--skip-access-check",
        action="store_true",
        help="略過啟動前 gated 權限檢查（不建議）",
    )
    return p.parse_args()


def _hf_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or None
    )


def check_gemma_access(model_id: str) -> None:
    """啟動前確認 gated 模型可存取；失敗時印出明確步驟後結束。"""
    token = _hf_token()
    if not token:
        print(
            "\n❌ 未設定 HF_TOKEN。\n"
            f"Gemma（{model_id}）是 gated 模型，沒有 token／未授權會 403。\n\n"
            "請依序：\n"
            f"  1. 開啟並同意條款：https://huggingface.co/{model_id}\n"
            "     （狀態若為 awaiting review，需等作者通過）\n"
            "  2. 建立 Read token：https://huggingface.co/settings/tokens\n"
            "  3. CMD：\n"
            "       set HF_TOKEN=hf_你的token\n"
            "     或 PowerShell：\n"
            "       $env:HF_TOKEN='hf_你的token'\n"
            "     （可選）寫入本機登入快取：\n"
            "       python -c \"from huggingface_hub import login; login(token='hf_你的token')\"\n"
            "  4. 再執行：python finetune_gema.py\n\n"
            "暫時無法通過審核時，可改跑：\n"
            "  python finetune_qwen3_4b.py\n"
        )
        raise SystemExit(2)

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    except Exception as e:
        print(f"⚠️ 無法匯入 huggingface_hub 做預檢：{e}（改由 transformers 載入時再報錯）")
        return

    try:
        hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            token=token,
        )
        print(f"✅ 已確認可存取 gated 模型：{model_id}")
    except GatedRepoError as e:
        print(
            "\n❌ Gemma gated 權限未通過（403）。\n"
            f"模型：{model_id}\n"
            f"詳情：{e}\n\n"
            "請到模型頁同意授權，並確認帳號狀態不是「awaiting review」：\n"
            f"  https://huggingface.co/{model_id}\n"
            "通過後用同一個帳號的 HF_TOKEN 再試。\n\n"
            "或改用免授權、本機已有快取的 Qwen：\n"
            "  python finetune_qwen3_4b.py\n"
        )
        raise SystemExit(3) from e
    except HfHubHTTPError as e:
        print(f"\n❌ 下載檢查失敗（HTTP）：{e}\n請確認網路與 HF_TOKEN。\n")
        raise SystemExit(4) from e
    except Exception as e:
        # 本機已有快取時，部分錯誤仍可能略過
        msg = str(e).lower()
        if "403" in msg or "gated" in msg:
            print(f"\n❌ 無法存取 {model_id}：{e}\n")
            raise SystemExit(3) from e
        print(f"⚠️ 預檢略過（稍後由 transformers 載入）：{e}")


def main():
    args = parse_args()
    model_id = (args.model or MODEL_ID).strip()
    data_path = resolve_data_path(args.data)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (Path(__file__).resolve().parent / output_root).resolve()

    print(f"[Gemma 微調] {model_id}")
    print("注意：此為 HF gated 模型，需授權 + HF_TOKEN。")
    if not args.skip_access_check:
        check_gemma_access(model_id)

    train_one(
        model_id=model_id,
        slug=args.slug,
        data_path=data_path,
        output_root=output_root,
        max_steps=args.max_steps,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        err = str(e)
        if "gated" in err.lower() or "403" in err or "awaiting a review" in err.lower():
            print(
                "\n❌ 仍是 Hugging Face gated 權限問題，不是訓練程式本身錯誤。\n"
                "請完成模型頁授權審核後再設 HF_TOKEN 重試；"
                "或改跑 python finetune_qwen3_4b.py\n"
            )
            sys.exit(3)
        raise
