"""
地端 LLM LoRA 微調（ISO 27001 / OT 資安）— 支援多基底模型以便比較。

路徑約定（皆相對此檔所在目錄自動解析，不必手動 cd）：
  資料集：    train_ai/train_llm/train.json
  checkpoint：train_ai/train_llm/outputs/<slug>/
  合併模型：  train_ai/models/<slug>/
  現有模型：  train_ai/train_llm/qwen_ot_merged_model、phi4_merged_model

用法：
  cd train_ai/train_llm
  set HF_TOKEN=你的token          # 可選；Llama 等需授權
  python train.py --list
  python train.py --preset qwen25-3b
  python train.py --preset phi4-mini
  python train.py --model Qwen/Qwen2.5-1.5B-Instruct --slug qwen25_1p5b_ot
  python train.py --presets qwen25-1.5b,phi4-mini   # 連續訓練多個以便比較
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_AI_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = TRAIN_AI_DIR.parent
MODELS_DIR = TRAIN_AI_DIR / "models"
PRESETS_PATH = SCRIPT_DIR / "model_presets.json"
DEFAULT_DATA = SCRIPT_DIR / "train.json"

# 常見 Causal LM LoRA 目標層（Qwen / Llama / Phi 系）
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def load_presets() -> dict:
    if not PRESETS_PATH.is_file():
        return {"presets": {}, "default_preset": "qwen25-3b"}
    with open(PRESETS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def slugify_model_id(model_id: str) -> str:
    name = model_id.split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name.lower().rstrip("_") + "_ot"


def resolve_data_path(data_arg: str) -> Path:
    p = Path(data_arg)
    if p.is_file():
        return p.resolve()
    cand = SCRIPT_DIR / data_arg
    if cand.is_file():
        return cand.resolve()
    cand2 = TRAIN_AI_DIR / data_arg
    if cand2.is_file():
        return cand2.resolve()
    raise FileNotFoundError(
        f"找不到訓練資料：{data_arg}\n"
        f"請確認存在：{DEFAULT_DATA}"
    )


def parse_args():
    preset_data = load_presets()
    default_preset = preset_data.get("default_preset", "qwen25-3b")

    parser = argparse.ArgumentParser(
        description="Fine-tune local LLM with LoRA（多基底可比較）"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出可用 preset 與已訓練模型後結束",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help=f"使用 model_presets.json 中的預設（預設建議：{default_preset}）",
    )
    parser.add_argument(
        "--presets",
        type=str,
        default=None,
        help="逗號分隔多個 preset，依序訓練以便比較，例如 qwen25-1.5b,phi4-mini",
    )
    parser.add_argument("--model", type=str, default=None, help="HuggingFace model id")
    parser.add_argument("--slug", type=str, default=None, help="輸出資料夾名稱（train_ai/models/<slug>）")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA), help="train.json 路徑")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(MODELS_DIR),
        help="合併模型輸出根目錄（預設 train_ai/models）",
    )
    return parser.parse_args()


def list_status():
    data = load_presets()
    presets = data.get("presets") or {}
    print("\n可用訓練 Preset：")
    for key, meta in presets.items():
        print(
            f"  - {key:14s}  {meta.get('model_id')}\n"
            f"                   → models/{meta.get('slug')} ｜ {meta.get('desc', '')}"
        )
    print(f"\n預設 preset：{data.get('default_preset')}")
    print(f"資料集：{DEFAULT_DATA} （存在={DEFAULT_DATA.is_file()}）")
    print(f"模型輸出根目錄：{MODELS_DIR}")
    if MODELS_DIR.is_dir():
        found = sorted(
            p.name for p in MODELS_DIR.iterdir()
            if p.is_dir() and (p / "config.json").is_file()
        )
        print("已訓練模型：" + (", ".join(found) if found else "（尚無）"))
    else:
        print("已訓練模型：（尚無，models 目錄未建立）")
    # 現有／相容路徑（實際模型多在 train_llm 下）
    legacy = [
        SCRIPT_DIR / "qwen_ot_merged_model",
        SCRIPT_DIR / "phi4_merged_model",
        PROJECT_ROOT / "qwen_ot_merged_model",
        PROJECT_ROOT / "phi4_merged_model",
        TRAIN_AI_DIR / "qwen_ot_merged_model",
        TRAIN_AI_DIR / "phi4_merged_model",
        TRAIN_AI_DIR / "outputs",
    ]
    legacy_ok = []
    for p in legacy:
        if p.is_dir() and (p / "config.json").is_file():
            legacy_ok.append(str(p))
        elif p.name == "outputs" and p.is_dir():
            legacy_ok.append(f"{p}（舊 LoRA checkpoint 根目錄）")
    if legacy_ok:
        print("現有／相容路徑：")
        for p in legacy_ok:
            print(f"  - {p}")
    print()


def train_one(
    model_id: str,
    slug: str,
    data_path: Path,
    output_root: Path,
    max_steps: int,
    max_length: int,
    batch_size: int,
):
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    ckpt_dir = SCRIPT_DIR / "outputs" / slug
    merged_dir = output_root / slug
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"基底模型 : {model_id}")
    print(f"資料集   : {data_path}")
    print(f"checkpoint: {ckpt_dir}")
    print(f"合併輸出 : {merged_dir}")
    print(f"max_steps={max_steps}, batch_size={batch_size}, max_length={max_length}")
    print("=" * 60)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=DEFAULT_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    system_prompt = (
        "你是專精於 OT 工控資安與 ISO 27001 合規稽核的 AI 專家。"
        "請全程以繁體中文回答，結構清楚，勿捏造數據。"
    )

    def format_prompts(examples):
        texts = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": output},
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                text = (
                    f"<|system|>\n{system_prompt}<|end|>\n"
                    f"<|user|>\n{instruction}<|end|>\n"
                    f"<|assistant|>\n{output}<|end|>"
                )
            texts.append(text)
        return {"text": texts}

    dataset = load_dataset("json", data_files=str(data_path), split="train")
    dataset = dataset.map(format_prompts, batched=True)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[c for c in dataset.column_names],
    )

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=max_steps,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        report_to="none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("開始 LoRA 微調...")
    trainer.train()

    # 另存 adapter，方便之後換基底重匯出
    adapter_dir = ckpt_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"LoRA adapter 已儲存：{adapter_dir}")

    print("合併 LoRA 權重...")
    merged_model = model.merge_and_unload()
    try:
        merged_model.to(
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    except Exception:
        pass

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))

    # 寫入訓練 meta，供 compare / app 識別
    meta = {
        "slug": slug,
        "base_model_id": model_id,
        "data": str(data_path),
        "max_steps": max_steps,
        "batch_size": batch_size,
        "merged_dir": str(merged_dir),
        "adapter_dir": str(adapter_dir),
    }
    with open(merged_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    cfg_path = merged_dir / "config.json"
    if cfg_path.is_file():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "quantization_config" in cfg:
            cfg.pop("quantization_config", None)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            print("已移除輸出模型的 quantization_config（利於全精度加速推論）")

    print(f"✅ 完整模型已儲存：{merged_dir}")
    print("比較模型：python compare_models.py --models " + slug)
    print("給 app 使用：set LLM_MODEL_PATH=" + str(merged_dir))
    print("（app.py 也會自動掃描 train_ai/models 與 train_ai/train_llm）")

    # 釋放顯存，方便連續訓練下一個 preset
    del trainer, model, merged_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_jobs(args) -> list[dict]:
    preset_data = load_presets()
    presets = preset_data.get("presets") or {}
    jobs = []

    if args.presets:
        keys = [x.strip() for x in args.presets.split(",") if x.strip()]
        for key in keys:
            if key not in presets:
                raise SystemExit(f"未知 preset：{key}（用 --list 查看）")
            meta = presets[key]
            jobs.append({
                "model_id": meta["model_id"],
                "slug": meta.get("slug") or slugify_model_id(meta["model_id"]),
                "max_steps": args.max_steps or int(meta.get("max_steps") or 200),
                "batch_size": args.batch_size or int(meta.get("batch_size") or 2),
            })
        return jobs

    if args.preset or (args.model is None and args.slug is None):
        key = args.preset or preset_data.get("default_preset", "qwen25-3b")
        if key not in presets:
            raise SystemExit(f"未知 preset：{key}（用 --list 查看）")
        meta = presets[key]
        return [{
            "model_id": args.model or meta["model_id"],
            "slug": args.slug or meta.get("slug") or slugify_model_id(meta["model_id"]),
            "max_steps": args.max_steps or int(meta.get("max_steps") or 200),
            "batch_size": args.batch_size or int(meta.get("batch_size") or 2),
        }]

    model_id = args.model
    if not model_id:
        raise SystemExit("請指定 --preset / --presets / --model")
    return [{
        "model_id": model_id,
        "slug": args.slug or slugify_model_id(model_id),
        "max_steps": args.max_steps or 200,
        "batch_size": args.batch_size or 2,
    }]


def main():
    args = parse_args()
    if args.list:
        list_status()
        return

    data_path = resolve_data_path(args.data)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (SCRIPT_DIR / output_root).resolve()

    jobs = build_jobs(args)
    print(f"將訓練 {len(jobs)} 個模型：{[j['slug'] for j in jobs]}")
    for job in jobs:
        train_one(
            model_id=job["model_id"],
            slug=job["slug"],
            data_path=data_path,
            output_root=output_root,
            max_steps=job["max_steps"],
            max_length=args.max_length,
            batch_size=job["batch_size"],
        )
    print("\n全部完成。可用下列指令比較：")
    slugs = ",".join(j["slug"] for j in jobs)
    print(f"  python compare_models.py --models {slugs}")


if __name__ == "__main__":
    main()
