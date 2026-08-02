"""
並排比較多個已微調（或基底）模型的回答，方便自行挑選。

用法：
  cd train_ai/train_llm
  python compare_models.py --list
  python compare_models.py --models qwen25_3b_ot,phi4_mini_ot
  python compare_models.py --models qwen25_3b_ot --also-base
  python compare_models.py --models qwen25_1p5b_ot,qwen25_3b_ot --question "什麼是 RADIUS？"
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_AI_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = TRAIN_AI_DIR.parent
MODELS_DIR = TRAIN_AI_DIR / "models"

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def discover_model_dirs() -> dict[str, Path]:
    """slug → path；含 train_ai/models/*、train_llm 現有模型與舊路徑。"""
    found: dict[str, Path] = {}

    def absorb(base: Path):
        if not base.is_dir():
            return
        for p in sorted(base.iterdir()):
            if p.is_dir() and (p / "config.json").is_file() and p.name not in found:
                found[p.name] = p

    absorb(MODELS_DIR)
    absorb(SCRIPT_DIR)  # train_ai/train_llm/*_merged_model（現有實際位置）

    legacy = {
        "qwen_ot_merged_model": [
            SCRIPT_DIR / "qwen_ot_merged_model",
            PROJECT_ROOT / "qwen_ot_merged_model",
            TRAIN_AI_DIR / "qwen_ot_merged_model",
        ],
        "phi4_merged_model": [
            SCRIPT_DIR / "phi4_merged_model",
            TRAIN_AI_DIR / "phi4_merged_model",
            PROJECT_ROOT / "phi4_merged_model",
        ],
    }
    for slug, cands in legacy.items():
        if slug in found:
            continue
        for c in cands:
            if (c / "config.json").is_file():
                found[slug] = c
                break
    return found


def resolve_model_path(name: str, catalog: dict[str, Path]) -> Path:
    if name in catalog:
        return catalog[name]
    p = Path(name)
    if (p / "config.json").is_file():
        return p.resolve()
    # 相對 models/
    cand = MODELS_DIR / name
    if (cand / "config.json").is_file():
        return cand
    raise FileNotFoundError(
        f"找不到模型「{name}」。可用 --list 查看，或傳完整路徑。"
    )


def load_pair(path: Path, hf_token: str | None):
    print(f"⏳ 載入：{path}")
    tok = AutoTokenizer.from_pretrained(str(path), token=hf_token, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        token=hf_token,
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def read_base_id(path: Path) -> str | None:
    meta = path / "train_meta.json"
    if meta.is_file():
        try:
            with open(meta, "r", encoding="utf-8") as f:
                return (json.load(f) or {}).get("base_model_id")
        except Exception:
            return None
    return None


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 512) -> str:
    system = (
        "你是專精於 OT 工控資安與 ISO 27001 合規稽核的 AI 專家。"
        "請以繁體中文簡潔回答。"
    )
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = (
            f"<|system|>\n{system}<|end|>\n"
            f"<|user|>\n{question}<|end|>\n"
            f"<|assistant|>\n"
        )

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return text.strip()


def parse_args():
    p = argparse.ArgumentParser(description="比較多個已訓練 LLM")
    p.add_argument("--list", action="store_true", help="列出可比較的本地模型")
    p.add_argument(
        "--models",
        type=str,
        default="",
        help="逗號分隔 slug 或路徑，例如 qwen25_3b_ot,phi4_mini_ot",
    )
    p.add_argument(
        "--also-base",
        action="store_true",
        help="若模型有 train_meta.json，一併載入其基底模型對照",
    )
    p.add_argument("--question", type=str, default="", help="單次問題（省略則進入互動）")
    p.add_argument("--max-new-tokens", type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()
    catalog = discover_model_dirs()

    if args.list or not args.models:
        print("\n可比較的本地模型（train_ai/models 與舊路徑）：")
        if not catalog:
            print("  （尚無）請先：python train.py --preset qwen25-3b")
        for slug, path in catalog.items():
            base = read_base_id(path) or "（未知基底）"
            print(f"  - {slug}\n      path: {path}\n      base: {base}")
        print("\n範例：python compare_models.py --models qwen25_3b_ot,phi4_mini_ot")
        if not args.models:
            return

    names = [x.strip() for x in args.models.split(",") if x.strip()]
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    loaded = []  # (label, model, tok)
    for name in names:
        path = resolve_model_path(name, catalog)
        model, tok = load_pair(path, hf_token)
        loaded.append((name, model, tok))
        if args.also_base:
            base_id = read_base_id(path)
            if base_id:
                label = f"BASE:{base_id.split('/')[-1]}"
                print(f"⏳ 載入基底：{base_id}")
                btok = AutoTokenizer.from_pretrained(
                    base_id, token=hf_token, trust_remote_code=True
                )
                if btok.pad_token is None:
                    btok.pad_token = btok.eos_token
                bmodel = AutoModelForCausalLM.from_pretrained(
                    base_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    token=hf_token,
                    trust_remote_code=True,
                )
                bmodel.eval()
                loaded.append((label, bmodel, btok))

    console = Console() if HAS_RICH else None
    print("\n" + "=" * 60)
    print(" 模型比較終端（輸入 exit 離開）")
    print(" 參與：", ", ".join(label for label, _, _ in loaded))
    print("=" * 60)

    def run_once(question: str):
        print("\n⏳ 各模型生成中...\n")
        for label, model, tok in loaded:
            ans = generate_answer(model, tok, question, args.max_new_tokens)
            if HAS_RICH:
                console.print(
                    Panel(
                        Markdown(ans or "（空回答）"),
                        title=f"[bold]{label}[/bold]",
                        border_style="cyan",
                        expand=True,
                    )
                )
                console.print()
            else:
                print("=" * 20 + f" {label} " + "=" * 20)
                print(ans)
                print()

    if args.question.strip():
        run_once(args.question.strip())
        return

    while True:
        q = input("\n💡 請輸入比較問題：").strip()
        if q.lower() in {"exit", "quit", "q"}:
            print("結束比較。")
            break
        if not q:
            continue
        run_once(q)


if __name__ == "__main__":
    main()
