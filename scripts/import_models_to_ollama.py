#!/usr/bin/env python3
"""
將 Semi-Shield 本地微調模型與 preset 基底匯入 Ollama。

用法（在專案根目錄）：
  python scripts/import_models_to_ollama.py
  python scripts/import_models_to_ollama.py --list
  python scripts/import_models_to_ollama.py --only qwen3_4b_ot,gemma_2b_ot
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

# HuggingFace model_id → Ollama 官方 tag（pull 用）
HF_TO_OLLAMA: dict[str, str] = {
    "Qwen/Qwen2.5-0.5B-Instruct": "qwen2.5:0.5b",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen2.5:1.5b",
    "Qwen/Qwen2.5-3B-Instruct": "qwen2.5:3b",
    "Qwen/Qwen2.5-7B-Instruct": "qwen2.5:7b",
    "Qwen/Qwen3-4B-Instruct-2507": "qwen3:4b",
    "Qwen/Qwen3-14B": "qwen3:14b",
    "microsoft/Phi-4-mini-instruct": "phi4",
    "google/gemma-2-2b-it": "gemma2:2b",
    "meta-llama/Llama-3.2-3B-Instruct": "llama3.2:3b",
}

LEGACY_SLUG_MAP: dict[str, str] = {
    "qwen_ot_merged_model": "qwen25_3b_ot",
    "phi4_merged_model": "phi4_mini_ot",
}

# 舊目錄名 → 匯入到 Ollama 的正式 slug
IMPORT_AS: dict[str, str] = {
    "qwen_ot_merged_model": "qwen25_3b_ot",
    "phi4_merged_model": "phi4_mini_ot",
}

# Ollama 無法正確載入 safetensors 的 slug → 改以基底 + SYSTEM 建立別名
SAFETENSORS_SKIP_SLUGS = frozenset({"gemma_2b_ot"})

# safetensors 匯入失敗時改用的 Ollama 基底 tag
FALLBACK_BASE: dict[str, str] = {
    "qwen3_4b_ot": "qwen3:4b",
    "qwen25_3b_ot": "qwen2.5:3b",
    "phi4_mini_ot": "phi4",
    "gemma_2b_ot": "gemma2:2b",
}


def find_ollama() -> str:
    exe = shutil.which("ollama")
    if exe:
        return exe
    for cand in (
        os.environ.get("OLLAMA_EXE", "").strip(),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
        r"C:\Program Files\Ollama\ollama.exe",
    ):
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
    return (path / "model.safetensors").is_file() and (path / "config.json").is_file()


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
        "qwen_ot_merged_model": [
            BASE_DIR / "train_ai" / "train_llm" / "qwen_ot_merged_model",
            BASE_DIR / "qwen_ot_merged_model",
        ],
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
        gb = (path / "model.safetensors").stat().st_size / (1024**3)
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

    # 額外：設定檔別名中的 Ollama tag
    if CONFIG_PATH.is_file():
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        for _alias, tag in (cfg.get("aliases") or {}).items():
            if ":" in str(tag) or str(tag).startswith(("qwen", "phi", "gemma", "llama")):
                pull_tags.add(str(tag).split(":")[0] + (":" + str(tag).split(":")[1] if ":" in str(tag) else ""))

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
        ("base_qwen3_4b", "qwen3:4b"),
        ("base_qwen25_3b", "qwen2.5:3b"),
        ("base_phi4_mini", "phi4"),
        ("base_gemma2_2b", "gemma2:2b"),
    ]
    for slug, tag in base_aliases:
        if only and slug not in only:
            continue
        create_base_wrapper(ollama, slug, tag, quantize=None)

    print(f"\n完成：成功 {ok}，失敗 {fail}")
    print("檢查：ollama list")
    print("啟動：run_ollama.bat  或  set OLLAMA_MODEL=qwen3_4b_ot && python app.py")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
