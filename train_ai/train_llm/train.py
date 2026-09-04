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
  python train.py --merge-only --slug qwen3_4b_ot --model Qwen/Qwen3-4B-Instruct-2507
      # 已有 LoRA 時：用全精度基底重新匯出（修補舊的殘缺 merge）
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
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
DEFAULT_DATA = (
    SCRIPT_DIR / "train.jsonl"
    if (SCRIPT_DIR / "train.jsonl").is_file()
    else SCRIPT_DIR / "train.json"
)

# 常見 Causal LM LoRA 目標層（Qwen / Llama / Phi 系）
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Gemma 4 多模態塔使用 Gemma4ClippableLinear；PEFT>=0.19 內建 LM-only 預設 regex。
# 若手動傳 DEFAULT_TARGET_MODULES 會誤命中 vision/audio 而失敗。
GEMMA4_LORA_TARGET_MODULES = None


def _read_model_config(source: str) -> dict:
    p = Path(source)
    if p.is_dir() and (p / "config.json").is_file():
        try:
            return json.loads((p / "config.json").read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _model_family(source: str, model_id: str = "") -> str:
    cfg = _read_model_config(source)
    model_type = str(cfg.get("model_type") or "").lower()
    archs = " ".join(str(a) for a in (cfg.get("architectures") or [])).lower()
    blob = f"{model_type} {archs} {model_id}".lower()
    if model_type == "gemma3_text" or "gemma3forcausallm" in archs:
        return "gemma3_text"
    if "gemma4" in blob or "multimodallm" in archs:
        return "gemma4"
    if "gemma3" in blob:
        return "gemma3"
    if "gemma" in blob:
        return "gemma"
    return "default"


def _bnb_config_for_family(family: str) -> BitsAndBytesConfig:
    compute = (
        torch.bfloat16
        if family in ("gemma3", "gemma4", "gemma")
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute,
        bnb_4bit_use_double_quant=True,
    )


def load_pretrained_for_training(
    base_source: str,
    *,
    hf_token: str | None,
    bnb_config: BitsAndBytesConfig | None = None,
    for_merge: bool = False,
    merge_device: str = "auto",
):
    """依 config 選擇 CausalLM / Gemma3 / MultimodalLM 載入。"""
    family = _model_family(base_source)
    common = dict(token=hf_token, trust_remote_code=True)
    extra: dict = {}
    if for_merge:
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        device_map = "auto"
        if merge_device == "cpu" or (
            merge_device == "auto" and not torch.cuda.is_available()
        ):
            device_map = None
        elif merge_device == "cuda":
            device_map = "auto"
        extra = dict(
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
    elif bnb_config is not None:
        extra = dict(
            quantization_config=bnb_config,
            device_map="auto",
        )
        if family in ("gemma3", "gemma3_text", "gemma4", "gemma"):
            extra["attn_implementation"] = "eager"
    else:
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        extra = dict(torch_dtype=dtype, device_map="auto", low_cpu_mem_usage=True)
        if family in ("gemma3", "gemma3_text", "gemma4", "gemma"):
            extra["attn_implementation"] = "eager"

    if family == "gemma4":
        try:
            from transformers import AutoModelForMultimodalLM

            return AutoModelForMultimodalLM.from_pretrained(
                base_source, **common, **extra
            )
        except ImportError as e:
            raise ImportError(
                "Gemma 4 需 transformers>=4.51；請 pip install -U transformers"
            ) from e

    if family == "gemma3_text":
        try:
            from transformers import Gemma3ForCausalLM

            return Gemma3ForCausalLM.from_pretrained(base_source, **common, **extra)
        except ImportError:
            pass

    if family == "gemma3":
        try:
            from transformers import Gemma3ForConditionalGeneration

            return Gemma3ForConditionalGeneration.from_pretrained(
                base_source, **common, **extra
            )
        except ImportError:
            pass

    return AutoModelForCausalLM.from_pretrained(base_source, **common, **extra)


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


def resolve_model_source(model_id: str, slug: str = "") -> str:
    """
    若 train_ai/models/<slug> 已有完整權重，優先用本地路徑載入基底（省頻寬）。
    回傳 HuggingFace id 或本地目錄字串。
    """
    candidates: list[Path] = []
    if slug:
        candidates.append(MODELS_DIR / slug)
        if slug.endswith("_ot"):
            base_slug = slug[: -len("_ot")]
            if base_slug:
                candidates.append(MODELS_DIR / base_slug)
    name = model_id.split("/")[-1].lower()
    for p in MODELS_DIR.iterdir() if MODELS_DIR.is_dir() else []:
        if not p.is_dir() or not (p / "config.json").is_file():
            continue
        if p.name.lower() == name or p.name.lower().replace("_ot", "") in name:
            candidates.append(p)
    seen: set[str] = set()
    for cand in candidates:
        key = str(cand.resolve())
        if key in seen:
            continue
        seen.add(key)
        if (cand / "config.json").is_file() and (
            (cand / "model.safetensors").is_file()
            or list(cand.glob("model-*.safetensors"))
            or list(cand.glob("pytorch_model*.bin"))
        ):
            try:
                cfg = json.loads((cand / "config.json").read_text(encoding="utf-8"))
                auto_map = cfg.get("auto_map") or {}
                for rel in auto_map.values():
                    if not isinstance(rel, str):
                        continue
                    mod_file = rel.split("::")[0].strip()
                    if mod_file and not mod_file.endswith(".py"):
                        mod_file = mod_file.split(".")[0] + ".py"
                    if mod_file and not (cand / mod_file).is_file():
                        print(
                            f"⚠️ 本地 {cand.name} 缺 {mod_file}，改從 HuggingFace 載入"
                        )
                        break
                else:
                    print(f"📂 使用本地基底：{cand}")
                    return str(cand.resolve())
            except Exception as e:
                print(f"⚠️ 讀取 {cand}/config.json 失敗（{e}），改從 HF 載入")
            continue
    return model_id


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
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="只存 LoRA adapter，不做全精度 merge（省顯存／時間）",
    )
    parser.add_argument(
        "--merge-device",
        type=str,
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="全精度 merge 時基底載入裝置（OOM 可改 cpu，較慢）",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="不訓練：只把既有 lora_adapter 用全精度基底重新 merge 匯出",
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


def _clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _export_full_precision_merge(
    *,
    model_id: str,
    adapter_dir: Path,
    merged_dir: Path,
    tokenizer,
    hf_token: str | None,
    merge_device: str = "auto",
) -> None:
    """
    正確匯出：用「全精度基底 + LoRA adapter」merge，再存檔。
    禁止在 4-bit QLoRA 權重上 merge_and_unload（會產出殘缺假全精度）。
    """
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    device_map = "auto"
    if merge_device == "cpu" or (
        merge_device == "auto" and not torch.cuda.is_available()
    ):
        device_map = None
    elif merge_device == "cuda":
        device_map = "auto"

    print("=" * 60)
    print("全精度 merge：重新載入基底（非 4-bit）+ 套用 LoRA 後再合併")
    print(f"  base   : {model_id}")
    print(f"  adapter: {adapter_dir}")
    print(f"  dtype  : {dtype}")
    print(f"  device : {merge_device} (device_map={device_map})")
    print("=" * 60)

    base = load_pretrained_for_training(
        model_id,
        hf_token=hf_token,
        for_merge=True,
        merge_device=merge_device,
    )
    if device_map is None:
        base = base.to("cpu")

    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    print("執行 merge_and_unload（全精度）...")
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()

    # 清掉舊的殘缺 safetensors，避免留下半套權重
    if merged_dir.is_dir():
        for pat in ("*.safetensors", "*.bin", "model.safetensors.index.json"):
            for old in merged_dir.glob(pat):
                try:
                    old.unlink()
                    print(f"已刪除舊權重：{old.name}")
                except Exception as e:
                    print(f"⚠️ 無法刪除 {old.name}：{e}")
    else:
        merged_dir.mkdir(parents=True, exist_ok=True)

    # 存檔前尽量放到 CPU，降低尖峰顯存
    try:
        merged_model = merged_model.to("cpu")
    except Exception:
        pass
    _clear_cuda()

    print(f"寫入完整模型 → {merged_dir}")
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))

    # 確認不應殘留 quantization_config
    cfg_path = merged_dir / "config.json"
    if cfg_path.is_file():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "quantization_config" in cfg:
            cfg.pop("quantization_config", None)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            print("已移除殘留 quantization_config")

    # 粗估大小，方便發現又存成殘缺檔
    total = sum(
        f.stat().st_size
        for f in merged_dir.glob("*.safetensors")
        if f.is_file()
    )
    print(f"合併權重合計：{total / 1e9:.2f} GB")
    if total < 1_000_000_000:
        print("⚠️ 權重異常偏小，請檢查 merge 是否成功")

    del peft_model, base, merged_model
    _clear_cuda()


def train_one(
    model_id: str,
    slug: str,
    data_path: Path,
    output_root: Path,
    max_steps: int,
    max_length: int,
    batch_size: int,
    skip_merge: bool = False,
    merge_device: str = "auto",
):
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    base_source = resolve_model_source(model_id, slug=slug)
    ckpt_dir = SCRIPT_DIR / "outputs" / slug
    merged_dir = output_root / slug
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"基底模型 : {base_source}")
    if base_source != model_id:
        print(f"  (HF id : {model_id})")
    print(f"資料集   : {data_path}")
    print(f"checkpoint: {ckpt_dir}")
    print(f"合併輸出 : {merged_dir}")
    print(f"max_steps={max_steps}, batch_size={batch_size}, max_length={max_length}")
    family = _model_family(base_source, model_id)
    train_mode = "bf16 LoRA" if family == "gemma4" else "QLoRA 4-bit（省顯存）"
    print(f"訓練方式 : {train_mode}")
    print(
        f"匯出方式 : "
        + ("只存 adapter（--skip-merge）" if skip_merge else "全精度基底+LoRA merge")
    )
    print("=" * 60)

    bnb_config = _bnb_config_for_family(family)
    load_kwargs: dict = dict(hf_token=hf_token, bnb_config=bnb_config)
    if family == "gemma4":
        # Gemma 4 多模態 + QLoRA 4-bit 與 PEFT 常不相容，改 bf16 LoRA
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        load_kwargs = dict(
            hf_token=hf_token,
            bnb_config=None,
            for_merge=False,
        )
        model = load_pretrained_for_training(
            base_source,
            **load_kwargs,
        )
        if hasattr(model, "vision_tower") and model.vision_tower is not None:
            for p in model.vision_tower.parameters():
                p.requires_grad = False
        if hasattr(model, "multi_modal_projector") and model.multi_modal_projector is not None:
            for p in model.multi_modal_projector.parameters():
                p.requires_grad = False
    else:
        model = load_pretrained_for_training(
            base_source,
            hf_token=hf_token,
            bnb_config=bnb_config,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        base_source,
        token=hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_targets = (
        GEMMA4_LORA_TARGET_MODULES if family == "gemma4" else DEFAULT_TARGET_MODULES
    )
    lora_kwargs: dict = dict(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if lora_targets is not None:
        lora_kwargs["target_modules"] = lora_targets
    peft_config = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, peft_config)

    system_prompt = (
        "你是專精於 OT 工控資安與 ISO 27001 合規稽核的 AI 專家。"
        "請全程以繁體中文回答，結構清楚，勿捏造數據。"
    )

    def _chat_template_supports_system() -> bool:
        """Gemma 等模板不支援 system role，需把系統提示併入 user。"""
        if not hasattr(tokenizer, "apply_chat_template"):
            return False
        try:
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "ping"},
                    {"role": "user", "content": "hi"},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            return True
        except Exception as e:
            msg = str(e).lower()
            if "system role" in msg or "system" in msg:
                print("ℹ️ 此模型 chat template 不支援 system role → 系統提示併入 user")
                return False
            # 其他錯誤先當不支援，避免訓練中途炸掉
            print(f"ℹ️ chat template 探測異常，改用不含 system 的格式：{e}")
            return False

    supports_system = _chat_template_supports_system()

    def _build_messages(instruction: str, output: str) -> list[dict]:
        if supports_system:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output},
            ]
        # Gemma：無 system → 併入第一則 user
        user_content = f"{system_prompt}\n\n{instruction}".strip()
        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]

    def _normalize_raw_messages(raw) -> list[dict]:
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, list):
            return []
        out: list[dict] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = (item.get("role") or "").strip().lower()
            content = (item.get("content") or "").strip()
            if role in ("user", "assistant", "system") and content:
                out.append({"role": role, "content": content})
        return out

    def _messages_for_template(raw_messages: list[dict]) -> list[dict]:
        """保留資料集 system；Gemma 等不支援 system 時併入 user。"""
        msgs = [dict(m) for m in raw_messages if m.get("content")]
        if supports_system:
            return msgs
        merged: list[dict] = []
        sys_parts = [m["content"] for m in msgs if m.get("role") == "system"]
        for m in msgs:
            role = m.get("role")
            if role == "system":
                continue
            if role == "user" and sys_parts and not any(
                x.get("role") == "user" for x in merged
            ):
                prefix = "\n\n".join(sys_parts).strip()
                merged.append({
                    "role": "user",
                    "content": f"{prefix}\n\n{m['content']}".strip(),
                })
            else:
                merged.append(dict(m))
        if not merged and sys_parts:
            merged.append({"role": "user", "content": sys_parts[0]})
        return merged

    def _render_chat(messages: list[dict], *, fallback_user: str = "", fallback_out: str = "") -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception as e:
                if "system" in str(e).lower():
                    fixed = _messages_for_template(messages)
                    return tokenizer.apply_chat_template(
                        fixed,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                raise
        return (
            f"<|user|>\n{fallback_user}<|end|>\n"
            f"<|assistant|>\n{fallback_out}<|end|>"
        )

    def format_prompts_legacy(examples):
        texts = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            messages = _build_messages(instruction, output)
            texts.append(_render_chat(messages, fallback_user=instruction, fallback_out=output))
        return {"text": texts}

    def format_prompts_messages(examples):
        texts = []
        for raw in examples["messages"]:
            messages = _messages_for_template(_normalize_raw_messages(raw))
            if not any(m.get("role") == "assistant" for m in messages):
                continue
            user_txt = "\n\n".join(
                m["content"] for m in messages if m.get("role") == "user"
            )
            asst_txt = next(
                (m["content"] for m in messages if m.get("role") == "assistant"),
                "",
            )
            texts.append(_render_chat(messages, fallback_user=user_txt, fallback_out=asst_txt))
        return {"text": texts}

    dataset = load_dataset("json", data_files=str(data_path), split="train")
    if "messages" in dataset.column_names:
        print(f"📚 資料格式：messages（{len(dataset)} 筆）")
        dataset = dataset.map(format_prompts_messages, batched=True)
    elif "instruction" in dataset.column_names and "output" in dataset.column_names:
        print(f"📚 資料格式：instruction/output（{len(dataset)} 筆）")
        dataset = dataset.map(format_prompts_legacy, batched=True)
    else:
        raise ValueError(
            "訓練資料需含 messages 或 instruction+output 欄位；"
            f"目前欄位：{dataset.column_names}"
        )
    dataset = dataset.filter(lambda x: bool((x.get("text") or "").strip()))
    print(f"📚 有效訓練樣本：{len(dataset)} 筆")

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

    # 1) 一定先存 adapter（這才是 QLoRA 的可靠產物）
    adapter_dir = ckpt_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"✅ LoRA adapter 已儲存：{adapter_dir}")

    # 寫 adapter 側 meta（即使 skip-merge 也有紀錄）
    adapter_meta = {
        "slug": slug,
        "base_model_id": model_id,
        "data": str(data_path),
        "max_steps": max_steps,
        "batch_size": batch_size,
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir),
        "train_mode": "qlora_4bit",
        "merge_mode": "skipped" if skip_merge else "full_precision_base_plus_lora",
    }
    with open(adapter_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(adapter_meta, f, ensure_ascii=False, indent=2)

    # 2) 釋放 4-bit 訓練模型，避免佔著顯存又去做錯誤 merge
    del trainer, model
    _clear_cuda()

    if skip_merge:
        print("已跳過全精度 merge（--skip-merge）。推論請用基底+adapter，或稍後再匯出。")
        print(f"adapter：{adapter_dir}")
        return

    # 3) 全精度基底 + adapter → 真正完整 merged 模型
    try:
        _export_full_precision_merge(
            model_id=model_id,
            adapter_dir=adapter_dir,
            merged_dir=merged_dir,
            tokenizer=tokenizer,
            hf_token=hf_token,
            merge_device=merge_device,
        )
    except Exception as e:
        print(f"❌ 全精度 merge 失敗：{e}")
        print(
            "LoRA adapter 仍可用。"
            "可改：python train.py ... --merge-device cpu"
            "，或只保留 adapter 稍後再匯出。"
        )
        raise

    meta = {
        **adapter_meta,
        "merge_mode": "full_precision_base_plus_lora",
        "merged_dir": str(merged_dir),
    }
    with open(merged_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ 完整全精度模型已儲存：{merged_dir}")
    print("比較模型：python compare_models.py --models " + slug)
    print("給 app 使用：set LLM_MODEL_PATH=" + str(merged_dir))
    print("（app.py 也會自動掃描 train_ai/models 與 train_ai/train_llm）")
    _clear_cuda()


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


def merge_only_one(
    *,
    model_id: str,
    slug: str,
    output_root: Path,
    merge_device: str = "auto",
) -> None:
    """已有 adapter 時，重新做全精度 merge（修補舊殘缺輸出）。"""
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    adapter_dir = SCRIPT_DIR / "outputs" / slug / "lora_adapter"
    if not (adapter_dir / "adapter_config.json").is_file():
        raise SystemExit(f"找不到 LoRA adapter：{adapter_dir}")

    # 優先從 adapter meta 讀基底
    meta_path = adapter_dir / "train_meta.json"
    if meta_path.is_file():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f) or {}
            model_id = meta.get("base_model_id") or model_id
        except Exception:
            pass

    merged_dir = output_root / slug
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_dir),
        token=hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=hf_token, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    _export_full_precision_merge(
        model_id=model_id,
        adapter_dir=adapter_dir,
        merged_dir=merged_dir,
        tokenizer=tokenizer,
        hf_token=hf_token,
        merge_device=merge_device,
    )
    meta = {
        "slug": slug,
        "base_model_id": model_id,
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir),
        "train_mode": "qlora_4bit",
        "merge_mode": "full_precision_base_plus_lora",
        "note": "merge-only re-export",
    }
    with open(merged_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"✅ merge-only 完成：{merged_dir}")


def main():
    args = parse_args()
    if args.list:
        list_status()
        return

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (SCRIPT_DIR / output_root).resolve()

    if args.merge_only:
        if not args.slug and not args.preset and not args.model:
            raise SystemExit("--merge-only 請指定 --slug（與可選 --model／--preset）")
        model_id = args.model
        slug = args.slug
        if args.preset or (not model_id and not slug):
            jobs = build_jobs(args)
            model_id = jobs[0]["model_id"]
            slug = jobs[0]["slug"]
        if not slug:
            raise SystemExit("--merge-only 需要 --slug")
        if not model_id:
            # 嘗試從既有 merged/adapter meta 推
            for cand in (
                SCRIPT_DIR / "outputs" / slug / "lora_adapter" / "train_meta.json",
                output_root / slug / "train_meta.json",
            ):
                if cand.is_file():
                    with open(cand, "r", encoding="utf-8") as f:
                        model_id = (json.load(f) or {}).get("base_model_id")
                    if model_id:
                        break
        if not model_id:
            raise SystemExit("--merge-only 需要 --model 或 train_meta 內的 base_model_id")
        merge_only_one(
            model_id=model_id,
            slug=slug,
            output_root=output_root,
            merge_device=str(args.merge_device or "auto"),
        )
        return

    data_path = resolve_data_path(args.data)
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
            skip_merge=bool(args.skip_merge),
            merge_device=str(args.merge_device or "auto"),
        )
    print("\n全部完成。可用下列指令比較：")
    slugs = ",".join(j["slug"] for j in jobs)
    print(f"  python compare_models.py --models {slugs}")


if __name__ == "__main__":
    main()
