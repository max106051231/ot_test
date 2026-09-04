#!/usr/bin/env python3
"""
將 Semi-Shield 本地微調模型與 preset 基底匯入 Ollama。

用法（在專案根目錄）：
  python scripts/import_models_to_ollama.py
  python scripts/import_models_to_ollama.py --list
  python scripts/import_models_to_ollama.py --only llama32_3b_ot,gemma_2b_ot
  python scripts/import_models_to_ollama.py --quantize q4_K_M
  python scripts/import_models_to_ollama.py --skip-pull

步驟：
  1. ollama pull 各 preset 對應的 Ollama 基底
  2. 對 train_ai/models/* 含 model.safetensors 的目錄：Modelfile FROM . → ollama create
  3. 其餘 preset（無本地權重）：以基底 + Semi-Shield SYSTEM 建立別名模型
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "train_ai" / "models"
OUTPUTS_DIR = BASE_DIR / "train_ai" / "train_llm" / "outputs"
TRAIN_PY = BASE_DIR / "train_ai" / "train_llm" / "train.py"
PRESETS_PATH = BASE_DIR / "train_ai" / "train_llm" / "model_presets.json"
CONFIG_PATH = BASE_DIR / "config" / "ollama_models.json"
MODELFILES_DIR = BASE_DIR / "scripts" / "ollama_modelfiles"

SYSTEM_PROMPT = (
    "你是 Semi-Shield Cyber Agent，專精於 OT 工控資安與 ISO 27001 合規稽核。"
    "請用繁體中文回答；引用控制項時標明 Annex A 編號；"
    "無日誌證據時勿臆測合規狀態（No Evidence, No Compliance Claim）。"
)

# HuggingFace model_id → Ollama 官方 tag（pull 用；不含 Qwen）
HF_TO_OLLAMA: dict[str, str] = {
    "microsoft/Phi-4-mini-instruct": "phi4",
    "google/gemma-2-2b-it": "gemma2:2b",
    "google/gemma-2-9b-it": "gemma2:9b",
    "meta-llama/Llama-3.2-3B-Instruct": "llama3.2:3b",
    "meta-llama/Llama-3.2-1B-Instruct": "llama3.2:1b",
    "meta-llama/Llama-3.1-8B-Instruct": "llama3.1:8b",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral:7b",
    "google/gemma-3-1b-it": "gemma3:1b-it-qat",
    "google/gemma-3-4b-it": "gemma3:4b-it-qat",
    "google/gemma-4-E2B-it-qat-q4_0-unquantized": "gemma4:e2b-it-qat",
    "google/gemma-4-E4B-it-qat-q4_0-unquantized": "gemma4:e4b-it-qat",
}

LEGACY_SLUG_MAP: dict[str, str] = {
    "phi4_merged_model": "phi4_mini_ot",
}

IMPORT_AS: dict[str, str] = {
    "phi4_merged_model": "phi4_mini_ot",
}

SAFETENSORS_SKIP_SLUGS = frozenset({"gemma_2b_ot", "gemma2_2b", "gemma2_9b"})

FALLBACK_BASE: dict[str, str] = {
    "phi4_mini_ot": "phi4",
    "gemma_2b_ot": "gemma2:2b",
    "gemma2_2b": "gemma2:2b",
    "gemma2_9b": "gemma2:9b",
    "llama32_3b_ot": "llama3.2:3b",
    "llama32_1b": "llama3.2:1b",
    "gemma3_1b_ot": "gemma3:1b-it-qat",
    "gemma3_4b_ot": "gemma3:4b-it-qat",
    "gemma4_e2b_ot": "gemma4:e2b-it-qat",
    "gemma4_e4b_ot": "gemma4:e4b-it-qat",
}


def find_ollama() -> str:
    exe = shutil.which("ollama")
    if exe:
        return exe
    candidates: list[str] = []
    if sys.platform == "win32":
        candidates.extend([
            os.environ.get("OLLAMA_EXE", "").strip(),
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
            r"C:\Program Files\Ollama\ollama.exe",
        ])
    else:
        candidates.extend([
            os.environ.get("OLLAMA_EXE", "").strip(),
            "/usr/local/bin/ollama",
            "/usr/bin/ollama",
            str(Path.home() / ".local" / "bin" / "ollama"),
        ])
    for cand in candidates:
        if cand and Path(cand).is_file():
            return str(Path(cand).resolve())
    raise FileNotFoundError(
        "找不到 ollama。請安裝 https://ollama.com/download 並確認 ollama serve 已啟動。"
    )


def run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"▶ {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def load_presets() -> list[dict]:
    if not PRESETS_PATH.is_file():
        return []
    data = json.loads(PRESETS_PATH.read_text(encoding="utf-8"))
    out: list[dict] = []
    for _key, meta in (data.get("presets") or {}).items():
        if not isinstance(meta, dict):
            continue
        slug = str(meta.get("slug") or "").strip()
        mid = str(meta.get("model_id") or "").strip()
        if slug and mid:
            out.append({"slug": slug, "model_id": mid, "desc": meta.get("desc") or ""})
    return out


def _has_merged_weights(path: Path) -> bool:
    """train_ai/models 目錄是否含可載入權重（含分片 safetensors）。"""
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    if any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin")):
        return True
    if (path / "model.safetensors.index.json").is_file():
        return True
    return False


def _has_lora_adapter(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and (
        (path / "adapter_model.safetensors").is_file()
        or (path / "adapter_model.bin").is_file()
    )


def discover_lora_adapters() -> dict[str, Path]:
    """train_ai/train_llm/outputs/<slug>/lora_adapter → slug。"""
    found: dict[str, Path] = {}
    if not OUTPUTS_DIR.is_dir():
        return found
    for p in sorted(OUTPUTS_DIR.iterdir()):
        if not p.is_dir():
            continue
        adapter = p / "lora_adapter"
        if _has_lora_adapter(adapter):
            found[p.name] = adapter.resolve()
    return found


def _merged_dir_from_meta(adapter_dir: Path, slug: str) -> Path:
    meta_path = adapter_dir / "train_meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            merged = Path(str(meta.get("merged_dir") or "").strip())
            if merged.is_dir():
                return merged.resolve()
        except Exception:
            pass
    return (MODELS_DIR / slug).resolve()


def merge_lora_adapter(slug: str, *, merge_device: str = "auto") -> Path | None:
    """LoRA（outputs）→ 全精度 merged（train_ai/models/<slug>）。"""
    merged_dir = MODELS_DIR / slug
    if _has_merged_weights(merged_dir):
        print(f"⏭️  已有 merged 權重：{merged_dir}")
        return merged_dir.resolve()

    lora = discover_lora_adapters().get(slug)
    if not lora:
        print(f"❌ 找不到 LoRA adapter：{OUTPUTS_DIR / slug / 'lora_adapter'}")
        return None

    print(f"\n=== 全精度 merge：{slug} ===")
    print(f"  LoRA   : {lora}")
    print(f"  輸出   : {merged_dir}")
    print("  （需 HF_TOKEN 若基底為 gated 模型，如 google/gemma-2-2b-it）")
    cmd = [
        sys.executable,
        str(TRAIN_PY),
        "--merge-only",
        "--slug",
        slug,
        "--merge-device",
        merge_device,
    ]
    try:
        run(cmd, cwd=TRAIN_PY.parent)
    except subprocess.CalledProcessError:
        print(f"❌ merge 失敗：{slug}")
        return None

    if _has_merged_weights(merged_dir):
        print(f"✅ merge 完成：{merged_dir}")
        return merged_dir.resolve()
    print(f"❌ merge 後仍找不到 model.safetensors：{merged_dir}")
    return None


def discover_local_merged() -> dict[str, Path]:
    """slug → 含 model.safetensors 的目錄。"""
    found: dict[str, Path] = {}
    if MODELS_DIR.is_dir():
        for p in sorted(MODELS_DIR.iterdir()):
            if not p.is_dir():
                continue
            if _has_merged_weights(p):
                found[p.name] = p.resolve()

    for slug, adapter in discover_lora_adapters().items():
        if slug in found:
            continue
        merged = _merged_dir_from_meta(adapter, slug)
        if _has_merged_weights(merged):
            found[slug] = merged

    # 舊路徑別名
    legacy_dirs = {
        "phi4_merged_model": [
            BASE_DIR / "train_ai" / "train_llm" / "phi4_merged_model",
            BASE_DIR / "phi4_merged_model",
        ],
    }
    for slug, cands in legacy_dirs.items():
        if slug in found:
            continue
        for c in cands:
            if (c / "model.safetensors").is_file():
                found[slug] = c.resolve()
                break
    return found


def ollama_has_model(ollama: str, name: str) -> bool:
    try:
        cp = run([ollama, "list"], check=False)
        base = name.split(":")[0].lower()
        for line in (cp.stdout or "").splitlines()[1:]:
            tag = (line.split()[0] if line.split() else "").lower()
            if tag == name.lower() or tag.split(":")[0] == base:
                return True
    except Exception:
        pass
    return False


def write_modelfile(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"📝 {path}")


def modelfile_safetensors(system: str = SYSTEM_PROMPT) -> str:
    return f"""FROM .
PARAMETER temperature 0
PARAMETER num_ctx 8192
SYSTEM \"\"\"{system}\"\"\"
"""


def modelfile_from_base(base_tag: str, system: str = SYSTEM_PROMPT) -> str:
    return f"""FROM {base_tag}
PARAMETER temperature 0
PARAMETER num_ctx 8192
SYSTEM \"\"\"{system}\"\"\"
"""


def import_safetensors(
    ollama: str,
    slug: str,
    model_dir: Path,
    *,
    quantize: str | None,
    fallback_base: str | None = None,
) -> bool:
    mf = model_dir / "Modelfile.ollama"
    write_modelfile(mf, modelfile_safetensors())
    cmd = [ollama, "create", slug, "-f", str(mf.name)]
    if quantize:
        cmd = [ollama, "create", "--quantize", quantize, slug, "-f", str(mf.name)]
    try:
        run(cmd, cwd=model_dir)
        print(f"✅ 已匯入微調模型：{slug} ← {model_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ safetensors 匯入失敗 {slug}：{e}")
        if fallback_base:
            print(f"   → 改以 Ollama 基底建立別名：{slug} ← {fallback_base}")
            return create_base_wrapper(ollama, slug, fallback_base, quantize=quantize)
        print(
            "   若為 Qwen3 等架構，Ollama 可能不支援直接 FROM safetensors；"
            "可手動 ollama create 或更新 Ollama 版本。"
        )
        return False


def create_base_wrapper(
    ollama: str,
    slug: str,
    base_tag: str,
    *,
    quantize: str | None,
    force: bool = False,
) -> bool:
    if not force and ollama_has_model(ollama, slug):
        print(f"⏭️  已存在，略過：{slug}")
        return True
    if force and ollama_has_model(ollama, slug):
        try:
            run([ollama, "rm", slug], check=False)
            print(f"🗑️  已移除舊模型：{slug}")
        except Exception:
            pass
    mf = MODELFILES_DIR / f"{slug}.Modelfile"
    write_modelfile(mf, modelfile_from_base(base_tag))
    cmd = [ollama, "create", slug, "-f", str(mf)]
    if quantize:
        cmd = [ollama, "create", "--quantize", quantize, slug, "-f", str(mf)]
    try:
        run(cmd, cwd=BASE_DIR)
        print(f"✅ 已建立基底別名：{slug} ← {base_tag}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 建立別名失敗 {slug}：{e}")
        return False


def pull_bases(ollama: str, tags: set[str]) -> None:
    for tag in sorted(tags):
        if ollama_has_model(ollama, tag):
            print(f"⏭️  基底已存在：{tag}")
            continue
        try:
            run([ollama, "pull", tag])
            print(f"✅ 已拉取：{tag}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  pull 失敗 {tag}：{e}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="匯入 Semi-Shield 模型到 Ollama")
    ap.add_argument("--list", action="store_true", help="只列出將匯入的模型")
    ap.add_argument("--only", type=str, default="", help="逗號分隔 slug，只處理指定模型")
    ap.add_argument("--skip-pull", action="store_true", help="略過 ollama pull 基底")
    ap.add_argument(
        "--quantize",
        type=str,
        default="",
        help="建立時量化，例如 q4_K_M（可縮小顯存／磁碟）",
    )
    ap.add_argument(
        "--skip-merge-lora",
        action="store_true",
        help="略過 LoRA→merged（outputs 僅 adapter 時不自動 merge）",
    )
    ap.add_argument(
        "--merge-device",
        type=str,
        default="auto",
        help="LoRA merge 裝置：auto / cuda / cpu",
    )
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else set()
    quantize = args.quantize.strip() or None
    merge_lora = not args.skip_merge_lora

    presets = load_presets()
    lora_only = discover_lora_adapters()
    local = discover_local_merged()

    if merge_lora and lora_only:
        merge_targets = only if only else set(lora_only.keys())
        for slug in sorted(merge_targets):
            if slug in local:
                continue
            if slug not in lora_only:
                continue
            merged_path = merge_lora_adapter(slug, merge_device=str(args.merge_device or "auto"))
            if merged_path:
                local[slug] = merged_path

    print("=== 本地微調（safetensors）===")
    for slug, path in sorted(local.items()):
        gb = sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / (1024**3)
        print(f"  {slug:22} {gb:5.1f} GB  {path}")
    if not local:
        print("  （無）")

    pending_lora = {s: p for s, p in lora_only.items() if s not in local}
    if pending_lora:
        print("\n=== 待 merge 的 LoRA（outputs）===")
        for slug, adapter in sorted(pending_lora.items()):
            print(f"  {slug:22}  {adapter}")
        print("  請先：python train_ai/train_llm/train.py --merge-only --slug gemma_2b_ot")
        print("  或執行：merge_gemma.bat")

    print("\n=== Preset 對照 ===")
    pull_tags: set[str] = set()
    wrapper_jobs: list[tuple[str, str]] = []
    for p in presets:
        slug = p["slug"]
        mid = p["model_id"]
        base = HF_TO_OLLAMA.get(mid, "")
        has_local = slug in local
        print(f"  {slug:22} base={base or '?':16} local={'是' if has_local else '否'}")
        if base:
            pull_tags.add(base)
        if not has_local and base and slug not in lora_only:
            wrapper_jobs.append((slug, base))

    official_tags = set(HF_TO_OLLAMA.values())
    # 額外：設定檔別名中的 Ollama 官方 tag（勿 pull 自訂 OT slug）
    if CONFIG_PATH.is_file():
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        for _alias, tag in (cfg.get("aliases") or {}).items():
            t = str(tag)
            if "qwen" in t.lower() or "qwen" in str(_alias).lower():
                continue
            if ":" in t or t in official_tags:
                pull_tags.add(t)

    if args.list:
        return 0

    try:
        ollama = find_ollama()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1

    print(f"\n🦙 Ollama：{ollama}\n")

    if not args.skip_pull and pull_tags:
        print("=== 拉取 Ollama 基底 ===")
        pull_bases(ollama, pull_tags)

    ok, fail = 0, 0

    print("\n=== 匯入本地微調 safetensors ===")
    imported_slugs: set[str] = set()
    for slug, path in sorted(local.items()):
        if only and slug not in only and IMPORT_AS.get(slug, slug) not in only:
            continue
        target = IMPORT_AS.get(slug, slug)
        if target in SAFETENSORS_SKIP_SLUGS:
            print(f"⚠️  {target}：略過 safetensors（Ollama 相容性），改建立基底別名")
            base_tag = FALLBACK_BASE.get(target, "")
            if base_tag and create_base_wrapper(ollama, target, base_tag, quantize=quantize):
                ok += 1
                imported_slugs.add(target)
            else:
                fail += 1
            continue
        if import_safetensors(
            ollama,
            target,
            path,
            quantize=quantize,
            fallback_base=FALLBACK_BASE.get(target),
        ):
            ok += 1
            imported_slugs.add(target)
            # 舊 slug 另建別名（FROM 剛匯入的模型）
            if slug in IMPORT_AS and slug != target:
                mf = MODELFILES_DIR / f"{slug}.Modelfile"
                write_modelfile(mf, modelfile_from_base(target))
                try:
                    run([ollama, "create", slug, "-f", str(mf)], cwd=BASE_DIR)
                    print(f"✅ 舊別名：{slug} ← {target}")
                except subprocess.CalledProcessError:
                    print(f"⚠️  舊別名略過：{slug}")
        else:
            fail += 1

    print("\n=== 建立 preset 基底別名（無本地權重者）===")
    for slug, base_tag in wrapper_jobs:
        if only and slug not in only:
            continue
        if slug in local or slug in IMPORT_AS.values() or slug in imported_slugs:
            continue
        if create_base_wrapper(ollama, slug, base_tag, quantize=quantize):
            ok += 1
        else:
            fail += 1

    # 常用 Ollama 基底 tag（供 UI 切換）
    print("\n=== 註冊微調前基底別名 base:* ===")
    base_aliases = [
        ("base_llama32_3b", "llama3.2:3b"),
        ("base_llama32_1b", "llama3.2:1b"),
        ("base_llama31_8b", "llama3.1:8b"),
        ("base_phi4_mini", "phi4"),
        ("base_gemma2_2b", "gemma2:2b"),
        ("base_gemma2_9b", "gemma2:9b"),
        ("base_mistral_7b", "mistral:7b"),
        ("gemma2_2b", "gemma2:2b"),
        ("gemma2_9b", "gemma2:9b"),
        ("llama32_1b", "llama3.2:1b"),
        ("base_gemma3_1b_qat", "gemma3:1b-it-qat"),
        ("base_gemma3_4b_qat", "gemma3:4b-it-qat"),
        ("base_gemma4_e2b_qat", "gemma4:e2b-it-qat"),
        ("base_gemma4_e4b_qat", "gemma4:e4b-it-qat"),
    ]
    for slug, tag in base_aliases:
        if only and slug not in only:
            continue
        create_base_wrapper(ollama, slug, tag, quantize=None)

    print(f"\n完成：成功 {ok}，失敗 {fail}")
    print("檢查：ollama list")
    print("啟動：run_ollama.bat  或  set OLLAMA_MODEL=llama32_3b_ot && python app.py")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
