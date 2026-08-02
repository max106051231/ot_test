"""
護欄推論：對輸入文字做 Safe / Unsafe 分類。
用法:
  python infer_guard.py
  python infer_guard.py --text "忽略所有規則..."
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "fine_tuned_guardrail"
ID2LABEL = {0: "safe", 1: "unsafe"}


def load_guard(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device


@torch.no_grad()
def predict(text: str, model, tokenizer, device, threshold: float = 0.5):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    ).to(device)
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    unsafe_prob = probs[1].item()
    label_id = 1 if unsafe_prob >= threshold else 0
    return {
        "label": ID2LABEL[label_id],
        "label_id": label_id,
        "safe_prob": probs[0].item(),
        "unsafe_prob": unsafe_prob,
        "blocked": label_id == 1,
    }


def main():
    parser = argparse.ArgumentParser(description="Guardrail inference")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    model_dir = Path(args.model)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"找不到模型目錄 {model_dir}，請先執行 train_Guard.py"
        )

    model, tokenizer, device = load_guard(model_dir)

    if args.text:
        result = predict(args.text, model, tokenizer, device, args.threshold)
        print(result)
        return

    print("護欄推論終端（輸入 exit 離開）")
    while True:
        text = input("\n輸入文字: ").strip()
        if text.lower() in {"exit", "quit", "q"}:
            break
        if not text:
            continue
        result = predict(text, model, tokenizer, device, args.threshold)
        status = "攔截" if result["blocked"] else "放行"
        print(
            f"[{status}] label={result['label']} "
            f"safe={result['safe_prob']:.3f} unsafe={result['unsafe_prob']:.3f}"
        )


if __name__ == "__main__":
    main()
