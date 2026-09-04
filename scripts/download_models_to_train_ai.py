#!/usr/bin/env python3
"""
將 HuggingFace 模型下載／複製到 train_ai/models/<slug>/。

用法（專案根目錄）：
  python scripts/download_models_to_train_ai.py
  python scripts/download_models_to_train_ai.py --list
  python scripts/download_models_to_train_ai.py --only llama32_3b_ot,phi4_mini_ot
  python scripts/download_models_to_train_ai.py --skip-existing

需 HF 授權的模型（Llama / Gemma 9B）請先：
  huggingface-cli login
  或 set HF_TOKEN=...
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "train_ai" / "models"
PRESETS_PATH = BASE_DIR / "train_ai" / "train_llm" / "model_presets.json"

# slug → HuggingFace model_id（不含 Qwen）
DEFAULT_TARGETS: list[tuple[str, str]] = [
    ("llama32_3b_ot", "meta-llama/Llama-3.2-3B-Instruct"),
    ("llama31_8b", "meta-llama/Llama-3.1-8B-Instruct"),
    ("llama32_1b", "meta-llama/Llama-3.2-1B-Instruct"),
    ("gemma2_9b", "google/gemma-2-9b-it"),
    ("gemma2_2b", "google/gemma-2-2b-it"),
    ("phi4_mini_ot", "microsoft/Phi-4-mini-instruct"),
    ("gemma3_1b", "google/gemma-3-1b-it"),
    ("gemma3_4b", "google/gemma-3-4b-it"),
    ("gemma4_e2b", "google/gemma-4-E2B-it-qat-q4_0-unquantized"),
    ("gemma4_e4b", "google/gemma-4-E4B-it-qat-q4_0-unquantized"),
]


def _hub_cache_roots() -> list[Path]:
    roots: list[Path] = []
    for key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE"):
        v = (os.environ.get(key) or "").strip()
        if v:
            roots.append(Path(v) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    out: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        key = str(r)
        if key not in seen and r.is_dir():
            seen.add(key)
            out.append(r)
    return out


def _cache_repo_dir(model_id: str) -> Path | None:
    folder = "models--" + model_id.replace("/", "--")
    for root in _hub_cache_roots():
        p = root / folder
        if p.is_dir():
            return p
    return None


def _latest_snapshot(repo_dir: Path) -> Path | None:
    snaps = repo_dir / "snapshots"
    if not snaps.is_dir():
        return None
    candidates = [p for p in snaps.iterdir() if p.is_dir() and (p / "config.json").is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _has_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "model.safetensors").is_file():
        return True
    if list(path.glob("model-*.safetensors")):
        return True
    if (path / "model.safetensors.index.json").is_file():
        return True
    return False


def _copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.is_symlink() or target.is_dir():
                if target.is_symlink():
                    target.unlink()
                elif target.exists():
                    shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _write_meta(slug: str, model_id: str, dest: Path) -> None:
    meta = {
        "slug": slug,
        "base_model_id": model_id,
        "source": "huggingface_download",
        "local_dir": str(dest.resolve()),
    }
    (dest / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def download_one(slug: str, model_id: str, *, skip_existing: bool) -> bool:
    dest = MODELS_DIR / slug
    ot_slug = f"{slug}_ot" if not slug.endswith("_ot") else slug
    ot_meta = MODELS_DIR / ot_slug / "train_meta.json"
    if ot_meta.is_file():
        try:
            meta = json.loads(ot_meta.read_text(encoding="utf-8"))
            if meta.get("merge_mode"):
                print(f"⏭️  略過下載（已有微調合併）：{ot_slug} → {MODELS_DIR / ot_slug}")
                return True
        except Exception:
            pass
    if skip_existing and _has_weights(dest):
        print(f"⏭️  已有權重，略過：{slug} → {dest}")
        return True

    # 1) 優先從本機 HF cache 複製
    repo = _cache_repo_dir(model_id)
    snap = _latest_snapshot(repo) if repo else None
    if snap and _has_weights(snap):
        print(f"📦 自 HF 快取複製：{model_id} → {dest}")
        if dest.exists():
            shutil.rmtree(dest)
        _copy_tree(snap, dest)
        _write_meta(slug, model_id, dest)
        print(f"✅ 完成（快取）：{slug}")
        return True

    # 2) 連網下載到 train_ai/models/<slug>
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ 請安裝 huggingface_hub：pip install huggingface_hub")
        return False

    print(f"⬇️  下載：{model_id} → {dest}")
    try:
        snapshot_download(
            model_id,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        _write_meta(slug, model_id, dest)
        print(f"✅ 完成：{slug}")
        return True
    except Exception as e:
        print(f"❌ 失敗 {slug} ({model_id})：{e}")
        if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
            print("   → 請至 HuggingFace 接受授權後執行：huggingface-cli login")
        return False


def load_targets() -> list[tuple[str, str]]:
    out = list(DEFAULT_TARGETS)
    if PRESETS_PATH.is_file():
        data = json.loads(PRESETS_PATH.read_text(encoding="utf-8"))
        seen = {s for s, _ in out}
        for _key, meta in (data.get("presets") or {}).items():
            if not isinstance(meta, dict):
                continue
            slug = str(meta.get("slug") or "").strip()
            mid = str(meta.get("model_id") or "").strip()
            if slug and mid and "qwen" not in slug.lower() and "qwen" not in mid.lower():
                if slug not in seen:
                    out.append((slug, mid))
                    seen.add(slug)
    return out


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="下載 HF 模型到 train_ai/models")
    ap.add_argument("--list", action="store_true", help="列出將下載的模型")
    ap.add_argument("--only", type=str, default="", help="逗號分隔 slug")
    ap.add_argument("--skip-existing", action="store_true", help="已有權重則略過")
    args = ap.parse_args()

    targets = load_targets()
    only = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else set()

    print(f"輸出目錄：{MODELS_DIR}\n")
    for slug, mid in targets:
        dest = MODELS_DIR / slug
        status = "已有" if _has_weights(dest) else "待下載"
        print(f"  {slug:22} {mid:42} [{status}]")

    if args.list:
        return 0

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, 0
    for slug, mid in targets:
        if only and slug not in only:
            continue
        if download_one(slug, mid, skip_existing=args.skip_existing):
            ok += 1
        else:
            fail += 1

    print(f"\n完成：成功 {ok}，失敗 {fail}")
    print(f"檢查：dir {MODELS_DIR}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
