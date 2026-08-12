import os
import re
import json
import gc
import threading
from pathlib import Path

# 離線優先：本地模型目錄存在時，禁止 Transformers / HF 連外網抓檔
from code.paths import project_root, web_dir, ot_logs_dir

_BASE_EARLY = project_root()


def _early_llm_backend() -> str:
    return (os.environ.get("LLM_BACKEND") or "ollama").strip().lower()


USE_OLLAMA = _early_llm_backend() in ("ollama", "1", "true", "yes")

# Ollama 模式略過 torch／transformers（Python 3.14 等環境可能無相容 wheel）
if USE_OLLAMA:
    os.environ.setdefault("OT_ENABLE_RAG", "1")
    os.environ.setdefault("ENABLE_GUARDRAIL", "1")
    os.environ.setdefault("ENABLE_AI_REVIEWER", "1")
    os.environ.setdefault("LLM_WARMUP", "0")

torch = None  # type: ignore[assignment,misc]
AutoModelForCausalLM = None  # type: ignore[misc,assignment]
AutoTokenizer = None  # type: ignore[misc,assignment]
StoppingCriteria = object  # type: ignore[misc,assignment]
StoppingCriteriaList = None  # type: ignore[misc,assignment]
BitsAndBytesConfig = None  # type: ignore[misc,assignment]

if not USE_OLLAMA:
    import torch as _torch
    torch = _torch
    from transformers import (
        AutoModelForCausalLM as _AutoModelForCausalLM,
        AutoTokenizer as _AutoTokenizer,
        StoppingCriteria as _StoppingCriteria,
        StoppingCriteriaList as _StoppingCriteriaList,
    )
    AutoModelForCausalLM = _AutoModelForCausalLM
    AutoTokenizer = _AutoTokenizer
    StoppingCriteria = _StoppingCriteria
    StoppingCriteriaList = _StoppingCriteriaList
    try:
        from transformers import BitsAndBytesConfig as _BitsAndBytesConfig
        BitsAndBytesConfig = _BitsAndBytesConfig
    except Exception:  # pragma: no cover
        BitsAndBytesConfig = None


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on", "edge")


def _local_llm_dirs_exist(base: Path) -> bool:
    checks = [
        base / "qwen_ot_merged_model" / "config.json",
        base / "phi4_merged_model" / "config.json",
        base / "train_ai" / "train_llm" / "qwen_ot_merged_model" / "config.json",
        base / "train_ai" / "train_llm" / "phi4_merged_model" / "config.json",
    ]
    if any(p.is_file() for p in checks):
        return True
    models_dir = base / "train_ai" / "models"
    if models_dir.is_dir():
        for p in models_dir.iterdir():
            if p.is_dir() and (p / "config.json").is_file():
                return True
    return False


def _detect_edge_mode() -> bool:
    """
    樹莓派／無 GPU 小裝置模式。
    - EDGE_MODE / LLM_EDGE=1 → 強制開啟
    - EDGE_MODE=0 → 強制關閉
    - 未設定：無 CUDA 時自動開啟
    """
    raw = os.environ.get("EDGE_MODE")
    if raw is None or str(raw).strip() == "":
        raw = os.environ.get("LLM_EDGE")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() in ("1", "true", "yes", "on", "edge")
    if USE_OLLAMA:
        return False
    return not torch.cuda.is_available()


EDGE_MODE = _detect_edge_mode()
# FORCE_CPU=1 等同 LLM_DEVICE=cpu（避開壞掉的 CUDA）
if _env_flag("FORCE_CPU", default=False):
    os.environ["LLM_DEVICE"] = "cpu"
LLM_DEVICE = (
    os.environ.get("LLM_DEVICE") or ("cpu" if EDGE_MODE else "auto")
).strip().lower()
# 執行期 CUDA 崩潰後設為 True，後續一律走 CPU
CUDA_DISABLED = False


def _early_is_hub_id(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+$", (value or "").strip()))


def _early_hf_hub_roots() -> list[Path]:
    roots: list[Path] = []
    for key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE"):
        v = (os.environ.get(key) or "").strip()
        if v:
            roots.append(Path(v))
    roots.append(Path.home() / ".cache" / "huggingface")
    out: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        for cand in (r, r / "hub"):
            key = str(cand)
            if key not in seen:
                seen.add(key)
                out.append(cand)
    return out


def _early_find_hf_snapshot(model_id: str) -> str | None:
    """啟動早期用：在本機 HF cache 找 snapshot（不連網）。"""
    if not _early_is_hub_id(model_id):
        return None
    repo = "models--" + model_id.replace("/", "--")
    for root in _early_hf_hub_roots():
        snaps = root / repo / "snapshots"
        if not snaps.is_dir():
            continue
        cands = sorted(
            [p for p in snaps.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for snap in cands:
            if (snap / "config.json").is_file():
                return str(snap.resolve())
    return None


def _early_hub_id_from_path(path: str) -> str | None:
    """路徑／snapshot → hub id（啟動早期可用）。"""
    if not path:
        return None
    if _early_is_hub_id(path):
        return path.strip()
    try:
        parts = Path(path).resolve().parts
    except Exception:
        parts = Path(path).parts
    for i, part in enumerate(parts):
        if part.startswith("models--") and "--" in part[8:]:
            return part[len("models--"):].replace("--", "/")
        if (
            re.fullmatch(r"[0-9a-f]{7,64}", part, re.I)
            and i >= 2
            and str(parts[i - 1]).lower() == "snapshots"
            and str(parts[i - 2]).startswith("models--")
        ):
            return parts[i - 2][len("models--"):].replace("--", "/")
    return None


def _pick_edge_default_model() -> str:
    """
    Edge/CPU 預設模型：優先本機已快取，避免離線去抓 HuggingFace。
    本機常見快取：Qwen2.5-3B-Instruct、Phi-4-mini。
    """
    forced = (os.environ.get("EDGE_LLM_MODEL") or "").strip()
    if forced:
        # 本機路徑
        p = Path(forced)
        if not p.is_absolute():
            p = _BASE_EARLY / p
        if (p / "config.json").is_file():
            return str(p.resolve())
        if _early_is_hub_id(forced):
            snap = _early_find_hf_snapshot(forced)
            return snap or forced
        return forced

    # 由小到大；有快取就用（你的環境已有 3B / Phi-4）
    candidates = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "microsoft/Phi-4-mini-instruct",
        "Qwen/Qwen3-4B-Instruct-2507",
    ]
    for mid in candidates:
        snap = _early_find_hf_snapshot(mid)
        if snap:
            print(f"🍊 Edge/CPU 採用本機 HF 快取：{mid}")
            return snap
    # 無快取時回 hub id（需 ALLOW_HF_DOWNLOAD=1 才會連網）
    return "Qwen/Qwen2.5-3B-Instruct"


EDGE_DEFAULT_MODEL = "" if USE_OLLAMA else _pick_edge_default_model()

# 正式機有本地微調模型 → 預設離線；Edge 有快取時也離線（避免誤連 HF）
_allow_hf_dl = _env_flag("ALLOW_HF_DOWNLOAD", default=False)
if not _allow_hf_dl:
    if _local_llm_dirs_exist(_BASE_EARLY) or EDGE_MODE or _early_find_hf_snapshot(
        "Qwen/Qwen2.5-3B-Instruct"
    ) or Path(EDGE_DEFAULT_MODEL).is_dir():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

def _configure_cpu_threads() -> int:
    """
    CPU 推論執行緒：
    - Edge／樹莓派：最多 4（避免過熱／搶 RAM）
    - Windows 桌機 FORCE_CPU：可用到 8（或 LLM_CPU_THREADS 覆寫）
    """
    if USE_OLLAMA or torch is None:
        return 1
    try:
        n = os.cpu_count() or 4
        if EDGE_MODE:
            threads = max(1, min(4, n - 1 if n > 1 else 1))
        else:
            threads = max(1, min(8, n))
        raw = (os.environ.get("LLM_CPU_THREADS") or "").strip()
        if raw.isdigit():
            threads = max(1, min(32, int(raw)))
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(threads))
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        print(f"🧵 CPU 執行緒：{threads}（可用 LLM_CPU_THREADS 調整）")
        return threads
    except Exception as e:
        print(f"⚠️ CPU 執行緒設定略過：{e}")
        return 1


# Edge／強制 CPU：預設關 RAG／護欄省 RAM與時間（可用環境變數重新開啟）
if EDGE_MODE or LLM_DEVICE == "cpu":
    os.environ.setdefault("OT_ENABLE_RAG", "0")
    os.environ.setdefault("ENABLE_GUARDRAIL", "0")
    os.environ.setdefault("LLM_WARMUP", "0")
    os.environ.setdefault("GUARDRAIL_DEVICE", "cpu")
    _configure_cpu_threads()

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from code.services.guardrail_service import guardrail_service
from code.services.compliance_service import (
    build_metrics_analysis,
    coverage_gaps,
    coverage_stats,
    load_controls,
    load_json as load_compliance_json,
)
from code.services.evidence_service import (
    register_control_bundle_evidence,
    list_evidence,
    get_evidence,
    traceability_stats,
)
from code.services.review_queue import enqueue_review, list_reviews, resolve_review, review_stats
from code.services.ai_reviewer_service import (
    ai_reviewer_enabled,
    reviewer_mode_summary,
    run_ai_reviewer,
)
from code.services.agent_orchestrator import workflow_spec, run_compliance_pipeline
from code.services import ollama_service

# RAG：啟用後會載入 embedding／索引，並在聊天與合規診斷時檢索
ENABLE_RAG = _env_flag("OT_ENABLE_RAG", default=not EDGE_MODE)
os.environ["OT_ENABLE_RAG"] = "1" if ENABLE_RAG else "0"

_rag_service = None
_rag_service_failed = False


def _get_rag_service():
    """延遲載入 RAG（避免 Ollama 模式／Python 3.14 強制 import numpy）。"""
    global _rag_service, _rag_service_failed
    if _rag_service_failed:
        return None
    if _rag_service is not None:
        return _rag_service
    if not ENABLE_RAG:
        return None
    try:
        from code.services.rag_service import rag_service as rs
        _rag_service = rs
        return _rag_service
    except Exception as e:
        _rag_service_failed = True
        print(f"⚠️ RAG 無法載入（已略過）：{e}")
        return None

BASE_DIR = str(project_root())
WEB_DIR = str(web_dir())
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "static"), static_url_path="/static")
CORS(app)

# =======================================================
# 護欄開關（False = 略過輸入攔截 / 輸出脫敏；True = 啟用）
# =======================================================
ENABLE_GUARDRAIL = _env_flag("ENABLE_GUARDRAIL", default=not EDGE_MODE)


def _rag_feature_enabled() -> bool:
    """功能開關 × 服務是否就緒。"""
    rs = _get_rag_service()
    return bool(ENABLE_RAG and rs is not None and getattr(rs, "enabled", False))


class _DisabledGuardrail:
    """護欄停用時的空實作，避免呼叫端報錯。"""
    mode = "disabled"

    def check_input(self, text):
        return {
            "label": "safe",
            "blocked": False,
            "safe_prob": 1.0,
            "unsafe_prob": 0.0,
            "mode": "disabled",
            "reason": "護欄已停用",
        }

    def sanitize_output(self, text):
        return text, False

    def block_message(self, result=None):
        return "護欄已停用"

    def mechanism_summary(self):
        return {
            "input_layers": ["（護欄已停用）"],
            "output_layers": ["（護欄已停用）"],
            "human_review": ["review_queue 仍可用於 LLM 診斷覆核"],
            "mode": "disabled",
            "threshold": None,
            "device": "n/a",
        }


if not ENABLE_GUARDRAIL:
    guardrail_service = _DisabledGuardrail()
    print("🛡️ 護欄功能已停用（ENABLE_GUARDRAIL=False）")

if EDGE_MODE:
    print(
        "🍊 Edge/CPU 模式：適合樹莓派等無 GPU 裝置 "
        f"（device={LLM_DEVICE}, default_model={EDGE_DEFAULT_MODEL}）"
    )

# =======================================================
# 全局變數與配置（速度優先；顯存不足時可設 LLM_4BIT=1）
# =======================================================
# 一律相對專案根目錄（不依賴啟動 cwd）
_ot_dir = ot_logs_dir()
OT_FOLDER = str(_ot_dir)


def _model_is_prequantized(path: str) -> bool:
    """本地／HF 路徑是否為 bitsandbytes 4-bit（CPU 無法載入）。"""
    p = Path(path)
    cfg_path = p / "config.json" if p.is_dir() else None
    if not cfg_path or not cfg_path.is_file():
        return False
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        qc = cfg.get("quantization_config") or {}
        return bool(qc.get("load_in_4bit") or qc.get("_load_in_4bit"))
    except Exception:
        return False


def _causal_weight_nbytes(path: Path) -> int:
    """合計因果 LM 權重檔大小（略過 LoRA adapter）。"""
    total = 0
    if not path.is_dir():
        return 0
    for pat in ("*.safetensors", "pytorch_model*.bin", "model*.bin"):
        for f in path.glob(pat):
            name = f.name.lower()
            if "adapter" in name:
                continue
            try:
                total += f.stat().st_size
            except Exception:
                pass
    return total


def _estimate_fp16_weight_nbytes(cfg: dict) -> int:
    """粗估 fp16/bf16 全量權重下限（用於抓不完整 merge）。"""
    h = int(cfg.get("hidden_size") or 0)
    i = int(cfg.get("intermediate_size") or (h * 4 if h else 0))
    layers = int(cfg.get("num_hidden_layers") or 0)
    vocab = int(cfg.get("vocab_size") or 0)
    if h <= 0 or layers <= 0:
        return 0
    # 近似：每層 attn(4h²) + MLP(≈3hi) + embed(vh)；tie 時 embed 仍計一次
    params = layers * (4 * h * h + 3 * h * i) + vocab * h
    return int(params * 2)  # fp16/bf16


def _read_train_meta(path: str | Path) -> dict:
    p = Path(path)
    meta_path = p / "train_meta.json" if p.is_dir() else None
    if not meta_path or not meta_path.is_file():
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _lora_adapter_dir(path: str | Path) -> Path | None:
    """微調目錄對應的 LoRA adapter（merge 殘缺時可改載 base+adapter）。"""
    p = Path(path)
    meta = _read_train_meta(p)
    cands: list[Path] = []
    ad = (meta.get("adapter_dir") or "").strip()
    if ad:
        cands.append(Path(ad))
    # 常見輸出位置
    slug = meta.get("slug") or p.name
    root = project_root()
    cands.append(root / "train_ai" / "train_llm" / "outputs" / slug / "lora_adapter")
    cands.append(p / "lora_adapter")
    for c in cands:
        try:
            if c.is_dir() and (c / "adapter_config.json").is_file() and (
                (c / "adapter_model.safetensors").is_file()
                or (c / "adapter_model.bin").is_file()
            ):
                return c.resolve()
        except Exception:
            continue
    return None


def _model_full_weights_ok(path: str) -> bool:
    """
    全量（非 4-bit）權重是否完整？
    4-bit 檔本來就小 → 視為格式 OK（另由 _model_is_prequantized 判斷裝置）。
    """
    p = Path(path)
    if not p.is_dir() or not (p / "config.json").is_file():
        return False
    if _model_is_prequantized(str(p)):
        return True
    try:
        with open(p / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return False
    nbytes = _causal_weight_nbytes(p)
    if nbytes < 400_000_000:  # <400MB：多半是 adapter／殘檔
        return False
    expect = _estimate_fp16_weight_nbytes(cfg)
    if expect > 0 and nbytes < int(expect * 0.55):
        print(
            f"⚠️ 偵測不完整 merge {p.name}："
            f"{nbytes/1e9:.2f}GB < 預估 fp16 的 55%（{expect/1e9:.2f}GB）"
            " → 強行載入會 MISMATCH"
        )
        return False
    return True


def _model_cpu_weights_ok(path: str) -> bool:
    """
    CPU 可載入的完整權重？
    - 4-bit → False
    - 權重遠小於 fp16 應有大小（如 4B 只剩 2.5GB）→ False（避免 MISMATCH）
    """
    if _model_is_prequantized(path):
        return False
    return _model_full_weights_ok(path)


def _llm_candidate_dirs(root: Path, *, edge: bool = False) -> list[Path]:
    """依優先序回傳可能的本地 LLM 目錄。"""
    train_ai = root / "train_ai"
    train_llm = train_ai / "train_llm"
    models_dir = train_ai / "models"
    if edge:
        # Edge：優先較小、非 4-bit 的本地成品
        preferred = [
            "qwen25_0p5b_ot",
            "qwen25_1p5b_ot",
            "qwen25_3b_ot",
            "qwen_ot_merged_model_copy",
            "qwen_ot_merged_model",
            "phi4_mini_ot",
            "phi4_merged_model",
        ]
    else:
        preferred = [
            "qwen25_3b_ot",
            "qwen_ot_merged_model",
            "qwen25_1p5b_ot",
            "phi4_mini_ot",
            "phi4_merged_model",
            "qwen25_7b_ot",
            "llama32_3b_ot",
        ]
    ordered: list[Path] = []

    if models_dir.is_dir():
        for name in preferred:
            ordered.append(models_dir / name)
        for p in sorted(models_dir.iterdir()):
            if p.is_dir() and p not in ordered:
                ordered.append(p)

    # 現有實際位置：train_ai/train_llm/*_merged_model
    for name in preferred:
        ordered.append(train_llm / name)
    if train_llm.is_dir():
        for p in sorted(train_llm.iterdir()):
            if p.is_dir() and (p / "config.json").is_file() and p not in ordered:
                ordered.append(p)

    # 相容：專案根 / train_ai 頂層舊路徑
    ordered.extend([
        root / "qwen_ot_merged_model",
        root / "phi4_merged_model",
        train_ai / "qwen_ot_merged_model",
        train_ai / "phi4_merged_model",
    ])
    return ordered


def _torch_cuda_available() -> bool:
    if USE_OLLAMA or torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _use_cuda_for_llm() -> bool:
    if USE_OLLAMA or torch is None:
        return False
    if CUDA_DISABLED:
        return False
    if LLM_DEVICE in ("cpu", "none"):
        return False
    if LLM_DEVICE in ("cuda", "gpu"):
        return _torch_cuda_available()
    return _torch_cuda_available()


def _try_enable_cuda_for_model(reason: str = "") -> bool:
    """
    模型無法在 CPU 載入（4-bit／殘缺 merge）時，嘗試改走 GPU。
    - 一般情況預設允許
    - FORCE_CPU=1 時需顯式設 LLM_ALLOW_GPU_FALLBACK=1
    """
    global CUDA_DISABLED, LLM_DEVICE
    if USE_OLLAMA or torch is None:
        return False
    if not _torch_cuda_available():
        print("⚠️ 無可用 CUDA，無法 GPU 後備")
        return False
    if _env_flag("FORCE_CPU", default=False):
        if not _env_flag("LLM_ALLOW_GPU_FALLBACK", default=False):
            print(
                "⚠️ FORCE_CPU=1，略過 GPU 後備"
                "（若要切微調 4-bit 模型可設 LLM_ALLOW_GPU_FALLBACK=1）"
            )
            return False
    elif not _env_flag("LLM_ALLOW_GPU_FALLBACK", default=True):
        print("⚠️ LLM_ALLOW_GPU_FALLBACK=0，略過 GPU 後備")
        return False
    CUDA_DISABLED = False
    LLM_DEVICE = "cuda"
    os.environ["LLM_DEVICE"] = "cuda"
    print(f"🔁 改以 GPU 載入模型… {(reason or '')[:160]}")
    return True


def _resolve_llm_model_path() -> str:
    """
    解析推論模型路徑（優先序）：
      1) 環境變數 LLM_MODEL_PATH
      2) CPU/Edge：完整非 4-bit 本地 → 本機 HF 快取（EDGE_DEFAULT）
      3) CUDA：train_ai 微調成品
    """
    env = (os.environ.get("LLM_MODEL_PATH") or "").strip()
    if env:
        if _is_hub_model_id(env):
            return env
        p = Path(env)
        if not p.is_absolute():
            p = Path(BASE_DIR) / p
        if (p / "config.json").is_file():
            resolved = str(p.resolve())
            if not _use_cuda_for_llm():
                if _model_is_prequantized(resolved) or not _model_cpu_weights_ok(resolved):
                    print(
                        f"⚠️ LLM_MODEL_PATH 不適合 CPU，改用 {EDGE_DEFAULT_MODEL}"
                    )
                    return EDGE_DEFAULT_MODEL
            return resolved
        print(f"⚠️ LLM_MODEL_PATH 無效或不含 config.json：{env}")

    root = Path(BASE_DIR)
    want_cpu = not _use_cuda_for_llm()

    # CPU/Edge：先用已驗證的 HF 快取（完整基座），再找本地完整 merge
    if want_cpu or EDGE_MODE:
        if Path(EDGE_DEFAULT_MODEL).is_dir() or _early_is_hub_id(EDGE_DEFAULT_MODEL):
            print(f"🍊 Edge/CPU：優先本機快取／預設 {EDGE_DEFAULT_MODEL}")
            # EDGE_DEFAULT 可能已是 snapshot 路徑
            if Path(EDGE_DEFAULT_MODEL).is_dir():
                return EDGE_DEFAULT_MODEL
        for p in _llm_candidate_dirs(root, edge=True):
            if not (p / "config.json").is_file():
                continue
            resolved = str(p.resolve())
            if not _model_cpu_weights_ok(resolved):
                continue
            print(f"🍊 Edge/CPU：採用完整本地模型 {p.name}")
            return resolved
        print(f"🍊 Edge/CPU：改用預設 {EDGE_DEFAULT_MODEL}")
        return EDGE_DEFAULT_MODEL

    for p in _llm_candidate_dirs(root, edge=False):
        if not (p / "config.json").is_file():
            continue
        return str(p.resolve())
    return str((root / "train_ai" / "train_llm" / "qwen_ot_merged_model").resolve())


def _is_hub_model_id(value: str) -> bool:
    """HuggingFace repo id，例如 Qwen/Qwen2.5-3B-Instruct。"""
    return bool(re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+$", (value or "").strip()))


MODEL_PATH = ""
LLM_BACKEND = (os.environ.get("LLM_BACKEND") or "ollama").strip().lower()
USE_OLLAMA = LLM_BACKEND in ("ollama", "1", "true", "yes")
if USE_OLLAMA:
    MODEL_PATH = ollama_service.resolve_model_name(
        os.environ.get("OLLAMA_MODEL") or ollama_service.default_model_name()
    )
    print(f"🦙 LLM 後端：Ollama | 預設模型={MODEL_PATH} | {ollama_service._base_url()}")
else:
    MODEL_PATH = _resolve_llm_model_path()
    print(f"🧠 LLM 後端：Transformers | 模型路徑：{MODEL_PATH}")
print(f"📁 OT 日誌目錄：{OT_FOLDER}")

# 速度檔：turbo | fast | balanced | edge（樹莓派／無 GPU）
# 可用環境變數覆寫：set LLM_SPEED=fast；CPU／Edge 未指定時預設 edge（勿用 GPU turbo 長輸出）
_speed_default = (
    "turbo"
    if USE_OLLAMA
    else ("edge" if (EDGE_MODE or LLM_DEVICE == "cpu") else "turbo")
)
SPEED_MODE = (os.environ.get("LLM_SPEED") or _speed_default).strip().lower()
if SPEED_MODE not in ("turbo", "fast", "balanced", "edge"):
    SPEED_MODE = _speed_default

if SPEED_MODE == "edge":
    # 樹莓派／CPU：短 prompt、短輸出，避免記憶體與時間爆掉
    MAX_INPUT_CHARS = 280
    MAX_NEW_TOKENS = 160
    MAX_NEW_TOKENS_AUDIT = 200
    MAX_NEW_TOKENS_VISUAL = 64
    MAX_PROMPT_TOKENS = 512
    MAX_OUTPUT_CHARS = 520
    MAX_OUTPUT_CHARS_REPORT = 720
    REPETITION_PENALTY = 1.05
    NO_REPEAT_NGRAM = 0
    RAG_CONTEXT_CHARS = 160
    CHAT_CTX_LIMIT = 140
elif SPEED_MODE == "turbo":
    # 體感速度優先；prompt 太短會截掉使用者問題（Qwen 尤易亂答）
    MAX_INPUT_CHARS = 400
    MAX_NEW_TOKENS = 320
    MAX_NEW_TOKENS_AUDIT = 400
    MAX_NEW_TOKENS_VISUAL = 96
    MAX_PROMPT_TOKENS = 896
    MAX_OUTPUT_CHARS = 900
    MAX_OUTPUT_CHARS_REPORT = 1800
    REPETITION_PENALTY = 1.08
    NO_REPEAT_NGRAM = 0
    RAG_CONTEXT_CHARS = 260
    CHAT_CTX_LIMIT = 220
elif SPEED_MODE == "fast":
    MAX_INPUT_CHARS = 520
    MAX_NEW_TOKENS = 220
    MAX_NEW_TOKENS_AUDIT = 480
    MAX_NEW_TOKENS_VISUAL = 120
    MAX_PROMPT_TOKENS = 720
    MAX_OUTPUT_CHARS = 820
    MAX_OUTPUT_CHARS_REPORT = 1260
    REPETITION_PENALTY = 1.06
    NO_REPEAT_NGRAM = 0
    RAG_CONTEXT_CHARS = 360
    CHAT_CTX_LIMIT = 300
else:
    MAX_INPUT_CHARS = 800
    MAX_NEW_TOKENS = 520
    MAX_NEW_TOKENS_AUDIT = 720
    MAX_NEW_TOKENS_VISUAL = 280
    MAX_PROMPT_TOKENS = 1280
    MAX_OUTPUT_CHARS = 1260
    MAX_OUTPUT_CHARS_REPORT = 1600
    REPETITION_PENALTY = 1.15
    NO_REPEAT_NGRAM = 4
    RAG_CONTEXT_CHARS = 700
    CHAT_CTX_LIMIT = 720

print(f"⚡ LLM 速度檔：{SPEED_MODE}（MAX_NEW_TOKENS={MAX_NEW_TOKENS}, AUDIT={MAX_NEW_TOKENS_AUDIT}）")


def _apply_cpu_speed_profile(reason: str = "") -> None:
    """
    CUDA 失效改走 CPU 後，把過長的 turbo 檔位壓成 edge 級，避免 3B CPU 生成數百 token。
    若使用者已明確設 LLM_SPEED=edge／fast 則只微調上限。
    """
    global MAX_NEW_TOKENS, MAX_NEW_TOKENS_AUDIT, MAX_NEW_TOKENS_VISUAL
    global MAX_PROMPT_TOKENS, CHAT_CTX_LIMIT, RAG_CONTEXT_CHARS, MAX_OUTPUT_CHARS
    global MAX_OUTPUT_CHARS_REPORT, SPEED_MODE

    _configure_cpu_threads()
    # 壓到 edge 預算（不強制改 SPEED_MODE 字串，避免誤導 UI）
    before = MAX_NEW_TOKENS
    MAX_NEW_TOKENS = min(int(MAX_NEW_TOKENS), 128)
    MAX_NEW_TOKENS_AUDIT = min(int(MAX_NEW_TOKENS_AUDIT), 160)
    MAX_NEW_TOKENS_VISUAL = min(int(MAX_NEW_TOKENS_VISUAL), 48)
    MAX_PROMPT_TOKENS = min(int(MAX_PROMPT_TOKENS), 512)
    CHAT_CTX_LIMIT = min(int(CHAT_CTX_LIMIT), 140)
    RAG_CONTEXT_CHARS = min(int(RAG_CONTEXT_CHARS), 160)
    MAX_OUTPUT_CHARS = min(int(MAX_OUTPUT_CHARS), 480)
    MAX_OUTPUT_CHARS_REPORT = min(int(MAX_OUTPUT_CHARS_REPORT), 640)
    if before != MAX_NEW_TOKENS:
        print(
            f"🍊 CPU 加速檔：MAX_NEW_TOKENS {before}→{MAX_NEW_TOKENS}"
            f"（{(reason or 'cpu')[:80]}）"
        )


# 設 LLM_4BIT=1 強制 4-bit（僅 CUDA）；CPU／Edge 會自動忽略
FORCE_4BIT = os.environ.get("LLM_4BIT", "").strip().lower() in ("1", "true", "yes")
# OT 監控掃描：預設只依檔案 mtime/size 失效（0=不強制逾時重掃）。
# 設 OT_CACHE_TTL>0 時，即使檔案未變超過 N 秒也會重掃（通常不必）。
OT_CACHE_TTL = float(os.environ.get("OT_CACHE_TTL") or "0")

# 本平台預設場域（可用 SITE_DOMAIN 覆寫）
SITE_DOMAIN = (os.environ.get("SITE_DOMAIN") or "半導體廠").strip() or "半導體廠"
print(f"🏭 預設場域：{SITE_DOMAIN}")

# 其他產業場域用詞 → 統一改寫，避免 RAG 範例把汽車廠等帶進回答
_FOREIGN_SITE_RE = re.compile(
    r"汽車組裝廠|汽車廠|車廠|食品廠|鋼鐵廠|石化廠|水泥廠|紙漿廠|"
    r"天然氣調壓站|水廠|港口碼頭|風力發電場|火力電廠|充電場站|"
    r"資料中心電力|醫院機電|製藥廠|面板廠|機場助航設施|BESS\s*場站",
    re.I,
)


def _sanitize_domain_text(text: str) -> str:
    """去掉／改寫異場域情境，鎖定本平台場域。"""
    if not text:
        return text
    t = text
    # 刪除「場域情境（xxx）：…」整段模板句
    t = re.sub(
        r"\**場域情境（[^）)]{1,40}）\**[：:][^\n]*",
        "",
        t,
    )
    t = _FOREIGN_SITE_RE.sub(SITE_DOMAIN, t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# 強制繁體中文輸出（所有 LLM 呼叫共用）
ZH_TW_OUTPUT_RULE = (
    "【語言硬性規定｜最高優先】你只能輸出「繁體中文」（zh-TW）。"
    "絕對禁止：日文、日本語、ひらがな、カタカナ、簡體中文、大段英文、全大寫英文牆。"
    "不可輸出日文結束語（如 答え終了），只能寫【回答結束】或【報告結束】。"
    "專有名詞（ISO 27001、RADIUS、SNMP、OOM、Linux）可保留原文，其餘說明必須繁中。"
    f"【場域硬性規定】本系統部署於「{SITE_DOMAIN}」。"
    f"所有案例、情境、建議都必須以{SITE_DOMAIN}（晶圓／封測／無塵室／製程設備／廠務）描述；"
    "禁止出現汽車組裝廠、食品廠、鋼鐵廠、石化廠等其他產業場域名稱。"
    "若參考資料含其他場域，請改寫為半導體廠用語，不要照抄。"
    "【長度】全文 ≤ 1260 字，寫完即停，禁止同一份報告用兩種語言各寫一遍。"
    "【禁令】禁止字母亂碼、禁止提示詞洩漏、禁止括號內的寫作指示（如字數提醒）。"
    "【禁令】不要把本段規定複述進回答（例如不要寫「請中文回答／使用 Markdown 格式」）。"
    "【排版】可用簡短標題與條列；禁止同一批條列詞彙反覆循環。"
)

# 一般對話用較短語言規則（減少 Phi 等模型照抄「請中文／Markdown」）
ZH_TW_CHAT_RULE = (
    "只用繁體中文回答，不要輸出日文或簡體。"
    "不要複述系統指示或寫「請用 Markdown」。"
    "禁止重複同一批條列（如一直重複「具體要求／具體情況」）；用完整句子說明重點。"
    f"情境以{SITE_DOMAIN}為準。"
)

# 合規報告：True＝以 LLM 自由撰寫為主（不同模型內容應有差異）
ENABLE_REPORT_LLM_FREEWRITE = True
# 三卡診斷報告版面（False＝一律自然對話／自由段落）
ENABLE_THREE_CARD_REPORT = False
# --- 暫時註解：日誌原文 grounded 診斷（會印「採用日誌原文 grounded…」並跳過 LLM）---
ENABLE_CISCO_GROUNDED_REPORT = False
# --- /暫時註解（恢復模板診斷時改 True）---

# 舊版強制三卡殼（freewrite 關閉時使用）
OUTPUT_FORMAT_RULE_STRICT = (
    "請【只輸出一次】下列繁體中文結構，不要再附日文或其他語言版本：\n"
    "## 地端 LLM 智慧合規診斷報告\n"
    "## 一、事件經過摘要\n"
    "（2-4 句繁體中文）\n"
    "## 二、不合規／風險分析\n"
    "- 重點 1\n"
    "- 重點 2\n"
    "## 三、具體修補建議\n"
    "- 建議 1\n"
    "- 建議 2\n"
    "【報告結束】"
)

# 自由撰寫：自然段落，禁止固定「一、二、三」章節殼
OUTPUT_FORMAT_RULE_FREE = (
    "請用繁體中文依本次日誌／量化摘要自由分析重點、風險與建議；"
    "不同控制項與不同模型應有不同論述；禁止套用固定罐頭句或無關 ICS 劇本。\n"
    "禁止固定章節模板：不要寫「地端 LLM 智慧合規診斷報告」，"
    "不要用「一、二、三」或「事件經過摘要／不合規分析／修補建議」等固定標題。\n"
    "可用自然段落或簡短條列；最後一行【報告結束】。"
)

OUTPUT_FORMAT_RULE = (
    ""
    if not ENABLE_THREE_CARD_REPORT
    else (
        OUTPUT_FORMAT_RULE_FREE
        if ENABLE_REPORT_LLM_FREEWRITE
        else OUTPUT_FORMAT_RULE_STRICT
    )
)

# 回答「格式不穩定」罐頭提示（False = 保留模型原文，不替換成建議改問）
ENABLE_UNSTABLE_FORMAT_FALLBACK = False

# 對話固定輸出（結論／說明／建議）— 已暫時關閉
# --- 暫時註解：固定「結論／說明／建議」格式功能 ---
ENABLE_FIXED_CHAT_FORMAT = False  # True 時啟用；呼叫端亦已註解 ensure_fixed_chat_format
# CHAT_OUTPUT_FORMAT_RULE = (
#     "請【只輸出一次】下列固定結構，不要增刪區塊標題：\n"
#     "## 回答\n\n"
#     "### 結論\n"
#     "（1-2 句繁體中文重點）\n\n"
#     "### 說明\n"
#     "- 要點 1\n"
#     "- 要點 2\n\n"
#     "### 建議\n"
#     "- 可執行步驟或下一步\n\n"
#     "【回答結束】\n"
#     "禁止輸出「地端 LLM 智慧合規診斷報告」或一／二／三卡舊格式。"
# )
CHAT_OUTPUT_FORMAT_RULE = ""  # 關閉期間留空，避免誤注入 prompt
# --- /暫時註解（恢復時還原上方 RULE，並 ENABLE_FIXED_CHAT_FORMAT=True）---


def format_fixed_chat_reply(
    conclusion: str,
    details=None,
    suggestions=None,
) -> str:
    """組出固定三段式對話回覆；關閉時改回一般條列。"""
    def _as_lines(items, fallback: str) -> list[str]:
        if items is None:
            return [f"- {fallback}"]
        if isinstance(items, str):
            text = items.strip()
            if not text:
                return [f"- {fallback}"]
            # 已是多行條列就保留；否則整段當一點
            lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
            out = []
            for ln in lines:
                s = ln.strip()
                if re.match(r"^\d+[\.、]\s+", s):
                    out.append(s)
                elif re.match(r"^[-*•]\s+", s):
                    out.append(re.sub(r"^[-*•]\s+", "- ", s))
                elif s.startswith("```") or s.startswith("#"):
                    out.append(s)
                else:
                    out.append(f"- {s}")
            return out or [f"- {fallback}"]
        out = []
        in_fence = False
        for it in items:
            s = str(it or "").rstrip()
            if not str(it or "").strip():
                continue
            st = s.strip()
            if st.startswith("```"):
                out.append(st)
                in_fence = not in_fence
                continue
            if in_fence:
                out.append(s)
                continue
            if re.match(r"^\d+[\.、]\s+", st):
                out.append(st)
            elif re.match(r"^[-*•]\s+", st):
                out.append(re.sub(r"^[-*•]\s+", "- ", st))
            else:
                out.append(f"- {st}")
        return out or [f"- {fallback}"]

    conc = (conclusion or "").strip() or "依目前資料整理如下。"
    # 結論保持單段文字，去掉多餘標題
    conc = re.sub(r"^#+\s*", "", conc).strip()
    conc = re.sub(r"^[-*•]\s+", "", conc).strip()

    # --- 暫時註解：固定三段式關閉時，輸出一般說明文字 ---
    if not ENABLE_FIXED_CHAT_FORMAT:
        parts = [
            conc,
            "",
            *_as_lines(details, "詳見監控／知識庫摘要。"),
            "",
            *_as_lines(suggestions, "若需更具體步驟，請補充 syslog 或控制項名稱。"),
            "",
            "【回答結束】",
        ]
        return "\n".join(parts)
    # --- /暫時註解 ---

    parts = [
        "## 回答",
        "",
        "### 結論",
        conc,
        "",
        "### 說明",
        *_as_lines(details, "詳見監控／知識庫摘要。"),
        "",
        "### 建議",
        *_as_lines(suggestions, "若需更具體步驟，請補充 syslog 或控制項名稱。"),
        "",
        "【回答結束】",
    ]
    return "\n".join(parts)


def ensure_fixed_chat_format(text: str, user_message: str = "") -> str:
    """
    將任意回覆收斂成固定「結論／說明／建議」格式。
    閒聊固定短答、純圖表補強可略過強制重組。
    """
    # --- 暫時註解：固定格式功能關閉 ---
    if not ENABLE_FIXED_CHAT_FORMAT:
        return (text or "").strip()
    # --- /暫時註解 ---
    raw = (text or "").strip()
    if not raw:
        return format_fixed_chat_reply(
            "目前沒有可回覆的內容。",
            ["請重新提問，或貼上 Cisco syslog。"],
            ["可問合規現況、修補步驟，或貼日誌分析。"],
        )

    # 已是標準格式（含三個區塊）→ 只保證結尾標記
    if (
        re.search(r"(?m)^##\s*回答\s*$", raw)
        and re.search(r"(?m)^###\s*結論\s*$", raw)
        and re.search(r"(?m)^###\s*說明\s*$", raw)
        and re.search(r"(?m)^###\s*建議\s*$", raw)
    ):
        body = re.sub(r"【(?:回答|報告)結束】", "", raw).strip()
        return body + "\n\n【回答結束】"

    # 保留 chart／mermaid 區塊，正文另行整形
    visual_blocks = re.findall(r"```(?:chart|mermaid)[\s\S]*?```", raw, flags=re.I)
    body = re.sub(r"```(?:chart|mermaid)[\s\S]*?```", "", raw, flags=re.I)
    body = re.sub(r"【(?:回答|報告)結束】", "", body).strip()

    # 去掉舊三卡標題殘留
    body = re.sub(
        r"^#{1,4}\s*地端\s*LLM.*?報告\s*$",
        "",
        body,
        flags=re.I | re.M,
    )
    body = re.sub(
        r"(?m)^#{1,4}\s*[一二三]、\s*(?:事件經過摘要|不合規.*|具體修補建議)\s*$",
        "",
        body,
    )
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    # 嘗試從既有 ### 結論／說明／建議 擷取
    def _section(name: str) -> str:
        m = re.search(
            rf"(?ms)^###\s*{name}\s*\n(.*?)(?=^###\s|\Z)",
            body,
        )
        return (m.group(1).strip() if m else "")

    sec_c = _section("結論")
    sec_d = _section("說明")
    sec_s = _section("建議")
    if sec_c or sec_d or sec_s:
        out = format_fixed_chat_reply(
            sec_c or "整理重點如下。",
            sec_d or body,
            sec_s or "可依需求再問細節或貼上 syslog。",
        )
    else:
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        # 抽出條列當說明／建議
        bullets = [
            re.sub(r"^[-*•]\s+", "", ln)
            for ln in lines
            if re.match(r"^[-*•]\s+", ln) or re.match(r"^\d+[\.、]\s+", ln)
        ]
        prose = [
            re.sub(r"^#+\s*", "", ln)
            for ln in lines
            if not re.match(r"^[-*•]\s+", ln)
            and not re.match(r"^\d+[\.、]\s+", ln)
            and not re.match(r"^#+\s*", ln)
        ]
        if not prose and lines:
            prose = [re.sub(r"^#+\s*", "", lines[0])]
        conclusion = prose[0] if prose else "依目前資料整理如下。"
        if len(conclusion) > 120:
            conclusion = conclusion[:120].rstrip("，,、；; ") + "…"
        details = bullets[:8] if bullets else (prose[1:6] if len(prose) > 1 else prose[:3] or [body[:240]])
        # 建議：優先取後段條列，否則給通用下一步
        suggestions = bullets[-3:] if len(bullets) >= 3 else (
            bullets if bullets else ["若需逐步操作指令或對照某控制項，請補充說明。"]
        )
        #  Harden／修補類：說明用全部步驟、建議用前幾步
        if wants_hardening_howto(user_message) or wants_remediation_steps(user_message):
            details = bullets[:10] if bullets else details
            suggestions = (
                ["依序執行上方步驟，完成後以 show／日誌複核。"]
                + (bullets[:2] if bullets else [])
            )
        out = format_fixed_chat_reply(conclusion, details, suggestions)

    if visual_blocks:
        out = out.replace(
            "【回答結束】",
            "\n\n".join(visual_blocks) + "\n\n【回答結束】",
        )
    return out

# 日文假名／常見日文標記（用於偵測與剔除）
_RE_HIRAGANA = re.compile(r"[\u3040-\u309F]")
_RE_KATAKANA = re.compile(r"[\u30A0-\u30FF]")
_RE_JP_MARKERS = re.compile(
    r"(答え終了|イベント|経過概要|安全対策|危険評価|修正提案|セキュリティ|"
    r"単語以内|單語以内|ひらがな|カタカナ|日本語|であります|です。|ます。|"
    r"サーバー|ログファイル|について)"
)

try:
    import zhconv
except ImportError:
    zhconv = None
    print("⚠️ 未安裝 zhconv，無法自動簡轉繁。請執行: pip install zhconv")

# 協議 → ISO 控制項 / 前端 key 對應
PROTOCOL_CONTROL_MAP = {
    "SNMP": {
        "ctrl_id": "A.8.24 (Crypto)",
        "label": "A.8.24 傳輸加密",
        "key": "sec_gem_log",
    },
    "SYSLOG": {
        "ctrl_id": "A.8.19 (Logs)",
        "label": "A.8.19 組態變更",
        "key": "recipe_audit",
    },
    "RADIUS": {
        "ctrl_id": "A.5.15 (Access)",
        "label": "A.5.15 存取控制",
        "key": "access_control",
    },
    "TACACS": {
        "ctrl_id": "A.5.15 (Access)",
        "label": "A.5.15 存取控制",
        "key": "access_control",
    },
}

CONTROL_TITLES = {
    "sec_gem_log": "A.8.24 密碼學與網絡傳輸安全校驗",
    "recipe_audit": "A.8.19 變更組態管理稽核",
    "access_control": "A.5.15 存取控制（OT 管理面登入／AAA 事件）",
    "patch_management": "A.8.8 技術弱點管理防禦日誌",
    "supplier_security": "A.5.19 供應商資安關係稽核預警",
    "malware_defense": "A.8.7 端點防範惡意軟體稽核",
}

# 減少碎片化；並開啟 TF32 / 高效注意力（Ampere+ / Blackwell）
if not USE_OLLAMA and torch is not None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            # flash／mem-efficient SDP 在 4-bit + 部分驅動／Py3.14 易 illegal access
            prefer_flash = os.environ.get("LLM_FLASH_SDP", "").strip().lower() in (
                "1", "true", "yes",
            )
            prefer_mem_eff = os.environ.get("LLM_MEM_EFF_SDP", "0").strip().lower() in (
                "1", "true", "yes",
            )
            torch.backends.cuda.enable_flash_sdp(prefer_flash)
            torch.backends.cuda.enable_mem_efficient_sdp(prefer_mem_eff)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass

# =======================================================
# 1. 模型載入與初始化（預設 bf16 加速；可退回 4-bit）
# =======================================================
if USE_OLLAMA:
    print("⏳ LLM 由 Ollama 管理（略過本地 Transformers 載入）…")
else:
    print("⏳ 正在載入工控/ISO27001 資安 LLM（速度模式）...")

_llm_lock = threading.RLock()
# 是否已做過「隱形先問你好」對話探測（丟掉冷啟動第一則）
_llm_chat_primed = False
# Gemma 等：chat template 是否支援 system role（None=尚未探測）
_llm_supports_system_role: bool | None = None
# 已載入模型：切換後釋放舊模型顯存（預設只保留 1 個）
# LLM_MODEL_CACHE_MAX>1 時可暫時多留，但仍建議 1
try:
    _LLM_CACHE_MAX = max(1, int(os.environ.get("LLM_MODEL_CACHE_MAX", "1") or "1"))
except Exception:
    _LLM_CACHE_MAX = 1
# 切換成功後是否強制卸載其他快取（預設是）
_LLM_RELEASE_ON_SWITCH = os.environ.get("LLM_RELEASE_ON_SWITCH", "1").strip().lower() not in (
    "0", "false", "no", "off",
)
_llm_cache: dict[str, dict] = {}
_llm_cache_lru: list[str] = []

_LLM_FRIENDLY_NAMES = {
    "qwen_ot_merged_model": "Qwen OT（微調後）",
    "phi4_merged_model": "Phi-4（微調後）",
    "phi4_lora_model": "Phi-4 LoRA",
    "qwen25_7b_ot": "Qwen2.5-7B OT（微調後）",
    "qwen3_4b_ot": "Qwen3-4B OT（微調後）",
    "qwen25_3b_ot": "Qwen2.5-3B OT（微調後）",
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B（Edge/CPU）",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B（微調前）",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B（微調前）",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B（微調前）",
    "Qwen/Qwen3-4B-Instruct-2507": "Qwen3-4B（微調前）",
    "microsoft/Phi-4-mini-instruct": "Phi-4 Mini（微調前）",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B（微調前）",
}

# 本地微調成品 → 對應微調前基底（無 train_meta 時的後備）
_FINETUNED_BASE_FALLBACK = {
    "qwen_ot_merged_model": "Qwen/Qwen2.5-3B-Instruct",
    "phi4_merged_model": "microsoft/Phi-4-mini-instruct",
}


def _normalize_llm_ref(value: str) -> str:
    """本地路徑 resolve；HF hub id 原樣保留。"""
    v = (value or "").strip()
    if not v:
        return v
    if _is_hub_model_id(v):
        return v
    return str(Path(v).resolve())


def _llm_friendly_name(slug: str) -> str:
    if slug in _LLM_FRIENDLY_NAMES:
        return _LLM_FRIENDLY_NAMES[slug]
    if _is_hub_model_id(slug):
        return f"{slug.split('/')[-1]}（微調前）"
    return slug.replace("_", " ").replace("-", " ")


def _llm_cache_touch(key: str) -> None:
    if key in _llm_cache_lru:
        _llm_cache_lru.remove(key)
    _llm_cache_lru.append(key)


def _cuda_ops_allowed() -> bool:
    """CUDA context 未宣告死亡時才允許呼叫 GPU API。"""
    return (not CUDA_DISABLED) and _torch_cuda_available()


def _cuda_empty_cache_quiet() -> None:
    if not _cuda_ops_allowed():
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _llm_unload_cache_entry(entry: dict | None, *, clear_active: bool = True) -> None:
    """釋放單一快取條目的 model／tokenizer 與 CUDA 顯存。"""
    global model, tokenizer
    if not entry:
        return
    print(f"🗑️ 卸載模型資源：{entry.get('slug') or entry.get('cache_key') or '?'}")
    if clear_active and entry.get("model") is model:
        model = None
        tokenizer = None
    try:
        del entry["model"]
        del entry["tokenizer"]
    except Exception:
        pass
    del entry
    gc.collect()
    _cuda_empty_cache_quiet()


def _llm_cache_evict_if_needed(keep_key: str | None = None) -> None:
    """超過快取上限時卸載最久未用的模型（保留 keep_key）。"""
    while len(_llm_cache) > _LLM_CACHE_MAX:
        victim = None
        for k in _llm_cache_lru:
            if k != keep_key and k in _llm_cache:
                victim = k
                break
        if not victim:
            break
        entry = _llm_cache.pop(victim, None)
        if victim in _llm_cache_lru:
            _llm_cache_lru.remove(victim)
        _llm_unload_cache_entry(entry, clear_active=True)


def _llm_cache_release_except(keep_key: str | None = None) -> None:
    """切換後釋放 keep_key 以外全部模型（確保舊模型顯存釋放）。"""
    for k in list(_llm_cache.keys()):
        if keep_key and k == keep_key:
            continue
        entry = _llm_cache.pop(k, None)
        if k in _llm_cache_lru:
            _llm_cache_lru.remove(k)
        # 作用中指標若指向新模型，勿清掉 global model
        clear_active = not (entry and entry.get("model") is model)
        _llm_unload_cache_entry(entry, clear_active=clear_active)


def _llm_cache_put(
    cache_key: str,
    m,
    tok,
    *,
    display_path: str,
    slug: str,
    stage: str,
) -> None:
    _llm_cache[cache_key] = {
        "model": m,
        "tokenizer": tok,
        "display_path": display_path,
        "slug": slug,
        "stage": stage,
        "cache_key": cache_key,
    }
    _llm_cache_touch(cache_key)
    _llm_cache_evict_if_needed(keep_key=cache_key)


def _llm_cache_activate(cache_key: str) -> bool:
    """從快取啟用模型；成功回 True。"""
    global model, tokenizer, MODEL_PATH
    entry = _llm_cache.get(cache_key)
    if not entry or entry.get("model") is None or entry.get("tokenizer") is None:
        return False
    model = entry["model"]
    tokenizer = entry["tokenizer"]
    MODEL_PATH = entry["display_path"]
    _llm_cache_touch(cache_key)
    return True


def _llm_cache_keys_info() -> list[dict]:
    out = []
    for k in _llm_cache_lru:
        e = _llm_cache.get(k) or {}
        out.append({
            "cache_key": k,
            "slug": e.get("slug"),
            "stage": e.get("stage"),
            "active": e.get("model") is model,
        })
    return out


def _load_base_plus_lora(base_id: str, adapter_dir: Path, *, use_cuda: bool):
    """載入 HF 基底 + LoRA adapter（用於 merge 殘缺的微調成品）。"""
    global MODEL_PATH
    try:
        from peft import PeftModel
    except Exception as e:
        raise RuntimeError(f"需要 peft 才能載入 LoRA adapter：{e}") from e

    load_base = _early_find_hf_snapshot(base_id) or base_id
    if _early_is_hub_id(load_base) and not _early_find_hf_snapshot(base_id):
        if not _env_flag("ALLOW_HF_DOWNLOAD", default=False):
            raise RuntimeError(
                f"基底「{base_id}」無本機快取，無法套用 LoRA。"
                "請設 ALLOW_HF_DOWNLOAD=1 下載，或先手動快取該模型。"
            )

    tok = AutoTokenizer.from_pretrained(str(adapter_dir))
    if tok.pad_token is None:
        # adapter 目錄可能沒完整 tokenizer → 改用基底
        try:
            tok = AutoTokenizer.from_pretrained(load_base)
        except Exception:
            pass
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

    if use_cuda:
        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        print(f"⏳ GPU 載入基底+LoRA：{base_id} + {adapter_dir}")
        # 4-bit 基底可省顯存；失敗再全精度
        base_m = None
        last_err = None
        if BitsAndBytesConfig is not None:
            try:
                base_m = AutoModelForCausalLM.from_pretrained(
                    load_base,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=compute_dtype,
                        bnb_4bit_use_double_quant=True,
                    ),
                    attn_implementation="eager",
                )
            except Exception as e:
                last_err = e
                print(f"⚠️ 基底 4-bit 載入失敗，改全精度：{e}")
        if base_m is None:
            try:
                base_m = AutoModelForCausalLM.from_pretrained(
                    load_base,
                    torch_dtype=compute_dtype,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                )
            except Exception as e:
                raise RuntimeError(f"基底載入失敗：{e}（先前：{last_err}）") from e
        m = PeftModel.from_pretrained(base_m, str(adapter_dir))
        try:
            m = m.merge_and_unload()
        except Exception as me:
            print(f"⚠️ LoRA merge_and_unload 略過（將以 PeftModel 推論）：{me}")
        m.eval()
        if hasattr(m, "config"):
            m.config.use_cache = True
        MODEL_PATH = str(adapter_dir.parent) if adapter_dir.parent.is_dir() else load_base
        print(f"✅ LLM 載入成功（GPU 基底+LoRA｜{base_id}）")
        return m, tok

    # CPU
    dtype_name = (os.environ.get("LLM_CPU_DTYPE") or "float32").strip().lower()
    if dtype_name in ("float16", "fp16", "half"):
        cpu_dtype = torch.float16
    elif dtype_name in ("bfloat16", "bf16"):
        cpu_dtype = torch.bfloat16
    else:
        cpu_dtype = torch.float32
    print(f"⏳ CPU 載入基底+LoRA：{base_id} + {adapter_dir}（較慢）")
    base_m = AutoModelForCausalLM.from_pretrained(
        load_base,
        torch_dtype=cpu_dtype,
        low_cpu_mem_usage=True,
        device_map=None,
        attn_implementation="eager",
    )
    base_m = base_m.to("cpu")
    m = PeftModel.from_pretrained(base_m, str(adapter_dir))
    try:
        m = m.merge_and_unload()
    except Exception as me:
        print(f"⚠️ LoRA merge_and_unload 略過：{me}")
    m = m.to("cpu")
    m.eval()
    if hasattr(m, "config"):
        m.config.use_cache = True
    if _env_flag("LLM_CPU_QUANT", default=True):
        try:
            m = torch.ao.quantization.quantize_dynamic(
                m, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("⚡ CPU 已套用動態 int8 量化（Linear）")
        except Exception as qe:
            print(f"⚠️ CPU int8 量化略過：{qe}")
    MODEL_PATH = str(adapter_dir.parent) if adapter_dir.parent.is_dir() else load_base
    print(f"✅ LLM 載入成功（CPU 基底+LoRA｜{base_id}）")
    return m, tok


def _load_llm(model_path: str | None = None):
    """
    載入策略：
    - CUDA：磁碟 4-bit / bf16 / LLM_4BIT=1
    - CPU／Edge：禁止 bitsandbytes；float32／float16 載入小模型
    - 本地 merge 不完整但有 LoRA：改載「基底 + adapter」（避免 MISMATCH）
    """
    global MODEL_PATH
    path = _normalize_llm_ref(model_path or MODEL_PATH)
    use_cuda = _use_cuda_for_llm()

    # 不完整 merge（如 qwen3_4b_ot 僅 2.6GB）→ 改 LoRA 或改用基底，勿硬載
    if (
        (not _early_is_hub_id(path))
        and Path(path).is_dir()
        and (not _model_is_prequantized(path))
        and (not _model_full_weights_ok(path))
    ):
        meta = _read_train_meta(path)
        base_id = (meta.get("base_model_id") or "").strip() or _FINETUNED_BASE_FALLBACK.get(
            Path(path).name, ""
        )
        adapter = _lora_adapter_dir(path)
        if adapter and base_id:
            print(
                f"⚠️ {Path(path).name} merge 不完整，改載基底+LoRA："
                f"{base_id} + {adapter.name}"
            )
            return _load_base_plus_lora(base_id, adapter, use_cuda=use_cuda)
        if base_id:
            snap = _early_find_hf_snapshot(base_id)
            if snap or _env_flag("ALLOW_HF_DOWNLOAD", default=False):
                print(
                    f"⚠️ {Path(path).name} merge 不完整且無可用 LoRA，"
                    f"改載基底 {base_id}"
                )
                path = snap or base_id
                MODEL_PATH = path
            else:
                raise RuntimeError(
                    f"「{Path(path).name}」權重不完整（會 MISMATCH），"
                    f"且基底「{base_id}」無本機快取。請重新完整 merge，"
                    "或先快取基底模型／確認 lora_adapter 存在。"
                )
        else:
            raise RuntimeError(
                f"「{Path(path).name}」權重不完整（會 MISMATCH），"
                "請改選完整模型或重新 merge。"
            )

    # CPU 無法載入 bitsandbytes 4-bit → 改用本機快取／Edge 預設
    if (not use_cuda) and (not _is_hub_model_id(path)) and _model_is_prequantized(path):
        print(
            f"⚠️ {Path(path).name} 為 CUDA 4-bit，CPU/Edge 改載入 {EDGE_DEFAULT_MODEL}"
        )
        path = EDGE_DEFAULT_MODEL
        MODEL_PATH = path

    # Hub id → 優先本機 snapshot，避免離線連網失敗（用 early helper，不依賴後方函式）
    if _early_is_hub_id(path):
        snap = _early_find_hf_snapshot(path)
        if snap:
            path = snap
            MODEL_PATH = path
        else:
            allow_dl = _env_flag("ALLOW_HF_DOWNLOAD", default=False)
            offline = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in (
                "1", "true", "yes",
            ) or os.environ.get("TRANSFORMERS_OFFLINE", "").strip().lower() in (
                "1", "true", "yes",
            )
            if offline or not allow_dl:
                raise RuntimeError(
                    f"模型「{path}」本機無 HF 快取，且禁止連網下載。"
                    "請改選本機已快取模型（如 Qwen2.5-3B-Instruct），"
                    "或設 ALLOW_HF_DOWNLOAD=1 後連網下載。"
                )

    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    pre_q = (not _is_hub_model_id(path)) and _model_is_prequantized(path)
    use_4bit = bool(use_cuda and BitsAndBytesConfig is not None and (FORCE_4BIT or pre_q))

    # ---------- CPU / Edge 路徑 ----------
    if not use_cuda:
        # ARM／樹莓派：float32 最穩；可設 LLM_CPU_DTYPE=float16 省 RAM（部分平台較慢）
        dtype_name = (os.environ.get("LLM_CPU_DTYPE") or "float32").strip().lower()
        if dtype_name in ("float16", "fp16", "half"):
            cpu_dtype = torch.float16
        elif dtype_name in ("bfloat16", "bf16"):
            cpu_dtype = torch.bfloat16
        else:
            cpu_dtype = torch.float32

        # 候選：先尊重指定路徑；失敗再退較小／本機快取
        # 若要強制從小模型起載：設 LLM_CPU_PREFER_SMALL=1
        prefer_small = _env_flag("LLM_CPU_PREFER_SMALL", default=False)
        cand_order: list[str] = []
        if prefer_small:
            cand_order.extend(
                [
                    "Qwen/Qwen2.5-0.5B-Instruct",
                    "Qwen/Qwen2.5-1.5B-Instruct",
                    EDGE_DEFAULT_MODEL,
                    path,
                    "Qwen/Qwen2.5-3B-Instruct",
                    "microsoft/Phi-4-mini-instruct",
                ]
            )
        else:
            cand_order.extend(
                [
                    path,
                    EDGE_DEFAULT_MODEL,
                    "Qwen/Qwen2.5-1.5B-Instruct",
                    "Qwen/Qwen2.5-0.5B-Instruct",
                    "Qwen/Qwen2.5-3B-Instruct",
                    "microsoft/Phi-4-mini-instruct",
                ]
            )
        path_candidates: list[str] = []
        for cand in cand_order:
            if not cand:
                continue
            c = cand
            if _early_is_hub_id(c):
                snap = _early_find_hf_snapshot(c)
                if not snap:
                    continue
                c = snap
            c = str(Path(c).resolve()) if Path(c).exists() else c
            if c not in path_candidates:
                if Path(c).is_dir() and not _model_cpu_weights_ok(c):
                    # HF 官方快取通常完整；僅擋本地殘缺 merge
                    if "huggingface" not in c.replace("\\", "/").lower():
                        continue
                path_candidates.append(c)

        last_err = None
        # CPU：Windows 桌機可先試 SDPA；失敗再 eager
        if EDGE_MODE:
            attn_try = (("eager", None), ("SDPA", "sdpa"))
        else:
            attn_try = (("SDPA", "sdpa"), ("eager", None))
        for try_path in path_candidates:
            # 換路徑時重載 tokenizer
            try:
                tok = AutoTokenizer.from_pretrained(try_path)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
            except Exception as te:
                last_err = te
                print(f"⚠️ tokenizer 載入失敗 {try_path}: {te}")
                continue
            for name, impl in attn_try:
                try:
                    kw = dict(
                        torch_dtype=cpu_dtype,
                        low_cpu_mem_usage=True,
                        device_map=None,
                    )
                    if impl:
                        kw["attn_implementation"] = impl
                    print(f"⏳ CPU 載入：{try_path}（{name}）")
                    m = AutoModelForCausalLM.from_pretrained(try_path, **kw)
                    m = m.to("cpu")
                    m.eval()
                    if hasattr(m, "config"):
                        m.config.use_cache = True
                    # 動態 int8：常見可再快一截（設 LLM_CPU_QUANT=0 關閉）
                    if _env_flag("LLM_CPU_QUANT", default=True):
                        try:
                            m = torch.ao.quantization.quantize_dynamic(
                                m, {torch.nn.Linear}, dtype=torch.qint8
                            )
                            print("⚡ CPU 已套用動態 int8 量化（Linear）")
                        except Exception as qe:
                            print(f"⚠️ CPU int8 量化略過：{qe}")
                    MODEL_PATH = try_path
                    print(
                        f"✅ LLM 載入成功（CPU {cpu_dtype} + {name}｜Edge={EDGE_MODE}）"
                    )
                    return m, tok
                except Exception as e1:
                    last_err = e1
                    msg = str(e1)
                    print(f"⚠️ CPU {name} 載入失敗，嘗試下一項：{msg[:240]}")
                    # 權重 shape 不符：換 attention 也沒用，直接換下一個模型
                    if (
                        "mismatch" in msg.lower()
                        or "size" in msg.lower()
                        or "shape" in msg.lower()
                    ):
                        break
        raise RuntimeError(f"CPU/Edge 無法載入 LLM：{last_err}")

    # ---------- CUDA 路徑 ----------
    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    def _from_pretrained(attn_impl, **extra):
        kw = dict(device_map="auto", low_cpu_mem_usage=True, **extra)
        if attn_impl:
            kw["attn_implementation"] = attn_impl
        return AutoModelForCausalLM.from_pretrained(path, **kw)

    def _attn_try_order(*, for_4bit: bool):
        """
        4-bit + SDPA/mem-efficient 在 Windows／新 PyTorch 易 warm-up illegal access。
        預設：4-bit → eager；全精度 → SDPA → eager；flash 僅 LLM_FLASH_ATTN=1。
        """
        force = (os.environ.get("LLM_ATTN") or "").strip().lower()
        if force in ("eager", "sdpa", "flash_attention_2", "flash"):
            if force in ("flash", "flash_attention_2"):
                return (("flash_attention_2", "flash_attention_2"), ("eager", None))
            if force == "sdpa":
                return (("SDPA", "sdpa"), ("eager", None))
            return (("eager", None),)
        order = []
        if os.environ.get("LLM_FLASH_ATTN", "").strip().lower() in ("1", "true", "yes"):
            order.append(("flash_attention_2", "flash_attention_2"))
        if for_4bit:
            order.extend((("eager", None), ("SDPA", "sdpa")))
        else:
            order.extend((("SDPA", "sdpa"), ("eager", None)))
        # 去重保序
        seen = set()
        out = []
        for item in order:
            if item[0] not in seen:
                seen.add(item[0])
                out.append(item)
        return tuple(out) or (("eager", None),)

    def _load_with_attn_fallback(*, for_4bit: bool = False, **extra):
        last = None
        for name, impl in _attn_try_order(for_4bit=for_4bit):
            try:
                return _from_pretrained(impl, **extra), name
            except Exception as e1:
                last = e1
                msg = str(e1)
                print(f"⚠️ {name} 載入失敗，嘗試下一項：{msg[:240]}")
                # 權重 shape MISMATCH：換 attention 也沒用
                low = msg.lower()
                if (
                    "mismatch" in low
                    or "ignore_mismatched_sizes" in low
                    or "size mismatch" in low
                ):
                    break
        raise RuntimeError(f"無法載入 LLM（attention 後備皆失敗）：{last}")

    loaded_4bit = False
    if use_4bit:
        extra = {}
        if not pre_q:
            extra["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        m, attn = _load_with_attn_fallback(for_4bit=True, **extra)
        loaded_4bit = True
        mode = "4-bit（磁碟已量化）" if pre_q else "4-bit NF4"
        print(f"✅ LLM 載入成功（{mode} + {attn}）")
    else:
        try:
            m, attn = _load_with_attn_fallback(
                for_4bit=False, torch_dtype=compute_dtype
            )
            print(f"✅ LLM 載入成功（{compute_dtype} 全精度 + {attn}，最快）")
        except Exception as e:
            if BitsAndBytesConfig is None:
                raise
            print(f"⚠️ 全精度載入失敗，改用 4-bit：{e}")
            m, attn = _load_with_attn_fallback(
                for_4bit=True,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                ),
            )
            loaded_4bit = True
            print(f"✅ LLM 載入成功（4-bit NF4 後備 + {attn}）")

    m.eval()
    if hasattr(m, "config"):
        m.config.use_cache = True

    # 可選：首次編譯後解碼更快（設 LLM_COMPILE=1 開啟；首次請求會較久）
    # 4-bit 預設不 compile（易與 bnb 衝突）
    if (
        (not loaded_4bit)
        and os.environ.get("LLM_COMPILE", "").strip().lower() in ("1", "true", "yes")
    ):
        try:
            m = torch.compile(m, mode="reduce-overhead")
            print("⚡ 已啟用 torch.compile（首次推論會暖機）")
        except Exception as e:
            print(f"⚠️ torch.compile 略過：{e}")

    return m, tok


def _is_cuda_fault(exc: BaseException) -> bool:
    msg = str(exc or "").lower()
    return any(
        k in msg
        for k in (
            "cuda error",
            "illegal memory access",
            "cudaerrorillegaladdress",
            "device-side assert",
            "cudnn_status",
            "cublas",
        )
    )


def _abandon_cuda(reason: str = "") -> None:
    """
    illegal access 後 CUDA context 已死：只設旗標，不再 synchronize／empty_cache
   （那些呼叫會一直噴同樣錯誤）。
    """
    global CUDA_DISABLED, LLM_DEVICE
    CUDA_DISABLED = True
    LLM_DEVICE = "cpu"
    os.environ["LLM_DEVICE"] = "cpu"
    print(f"🧯 放棄 CUDA（後續僅 CPU）… {(reason or '')[:160]}".strip())
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass
    gc.collect()


def _cuda_recover(reason: str = "") -> None:
    """輕量清理；若已判定 CUDA 死亡則只 abandon，不碰 GPU API。"""
    if CUDA_DISABLED or _is_cuda_fault(Exception(reason or "")):
        _abandon_cuda(reason)
        return
    print(f"🧯 CUDA 恢復中… {(reason or '')[:160]}".strip())
    gc.collect()
    if not _cuda_ops_allowed():
        return
    # 非 illegal-access 的 OOM 等：可試 empty_cache；勿强制 synchronize
    _cuda_empty_cache_quiet()
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def _failover_llm_to_cpu(reason: str = "") -> bool:
    """
    CUDA illegal access 後 GPU context 通常已壞死，再重試 GPU 無用。
    卸載 GPU 模型，改載 CPU 輕量模型（本機 HF 快取）。
    """
    global model, tokenizer, MODEL_PATH, CUDA_DISABLED, LLM_DEVICE, EDGE_MODE
    global _llm_chat_primed

    if CUDA_DISABLED and model is not None and tokenizer is not None:
        try:
            if str(next(model.parameters()).device).startswith("cpu"):
                return True
        except Exception:
            pass

    print(f"🔁 CUDA 已失效，切換 CPU 後備模型… {(reason or '')[:140]}")
    # 先宣告放棄 CUDA，避免卸載／清理時再呼叫 GPU API
    _abandon_cuda(reason)

    # 不要清掉離線旗標；改用本機 HF 快取（除非明確允許下載）
    if _env_flag("ALLOW_HF_DOWNLOAD", default=False):
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    try:
        for k in list(_llm_cache.keys()):
            entry = _llm_cache.pop(k, None)
            if k in _llm_cache_lru:
                _llm_cache_lru.remove(k)
            _llm_unload_cache_entry(entry, clear_active=True)
    except Exception as e:
        print(f"⚠️ 卸載 GPU 快取略過：{e}")

    # 丟掉已中毒的 GPU 模型（解構子若再噴 CUDA 錯誤則忽略）
    dead_m, dead_t = model, tokenizer
    model = None
    tokenizer = None
    try:
        del dead_m, dead_t
    except Exception:
        pass
    gc.collect()

    # 重新挑選有快取的 CPU 模型（避免抓不存在的 0.5B）
    fallback = _pick_edge_default_model()
    MODEL_PATH = fallback
    try:
        new_m, new_t = _load_llm(fallback)
        if new_m is None or new_t is None:
            raise RuntimeError("CPU 後備載入回傳空值")
        model, tokenizer = new_m, new_t
        cache_key = _normalize_llm_ref(fallback)
        hub_slug = _early_hub_id_from_path(str(fallback)) or (
            Path(fallback).name if Path(fallback).is_dir() else str(fallback)
        )
        _llm_cache_put(
            cache_key,
            new_m,
            new_t,
            display_path=hub_slug if _early_is_hub_id(hub_slug) else fallback,
            slug=f"base:{hub_slug}" if _early_is_hub_id(hub_slug) else hub_slug,
            stage="base",
        )
        print(f"✅ CPU 後備模型就緒：{fallback}")
        _llm_chat_primed = False
        try:
            _apply_cpu_speed_profile("cuda→cpu failover")
        except Exception as pe:
            print(f"⚠️ CPU 速度檔套用略過：{pe}")
        return True
    except Exception as e:
        print(f"❌ CPU 後備模型載入失敗：{e}")
        model, tokenizer = None, None
        return False


model = None
tokenizer = None
if USE_OLLAMA:
    ollama_service.init(model=MODEL_PATH)
    MODEL_PATH = ollama_service.current_model_name()
else:
    try:
        model, tokenizer = _load_llm()
    except Exception as e:
        print(f"❌ 模型載入失敗，請確認 {MODEL_PATH} 路徑是否正確：{str(e)}")
        model, tokenizer = None, None
        # CUDA 載入階段就爆 → 直接 CPU 後備
        if _is_cuda_fault(e) or "cuda" in str(e).lower():
            _failover_llm_to_cpu(str(e))

# 啟動暖機（Edge 預設關閉；CUDA 預設開啟；Ollama 可選）
_warmup_default = "0" if (EDGE_MODE or USE_OLLAMA) else "1"
if USE_OLLAMA:
    if os.environ.get("LLM_WARMUP", _warmup_default).strip().lower() not in (
        "0", "false", "no",
    ):
        try:
            ollama_service.prime_chat()
            print("🔥 Ollama 暖機完成")
        except Exception as _we:
            print(f"⚠️ Ollama 暖機失敗：{_we}")
elif (
    model is not None
    and tokenizer is not None
    and os.environ.get("LLM_WARMUP", _warmup_default).strip().lower()
    not in ("0", "false", "no")
):
    try:
        _w_dev = next(model.parameters()).device
        _w_ids = tokenizer("OK", return_tensors="pt")
        _w_ids = {k: v.to(_w_dev) for k, v in _w_ids.items()}
        # 4-bit／CUDA：關閉 KV cache 較不易踩到壞掉的 SDP kernel
        _w_use_cache = (not str(_w_dev).startswith("cuda"))
        with torch.inference_mode():
            model.generate(
                **_w_ids,
                max_new_tokens=1,
                do_sample=False,
                use_cache=_w_use_cache,
                pad_token_id=tokenizer.pad_token_id,
            )
        del _w_ids
        _cuda_empty_cache_quiet()
        print("🔥 LLM 暖機完成（後續回答會較快）")
    except Exception as _we:
        print(f"⚠️ LLM 暖機失敗：{_we}")
        if _is_cuda_fault(_we):
            print("⚠️ 啟動暖機即 CUDA 崩潰 → 立即切 CPU 後備")
            _failover_llm_to_cpu(str(_we))

def _prime_llm_chat(*, force: bool = False) -> bool:
    """
    在使用者第一則問題前，先用 chat template 問一次「你好」並丟棄回覆。
    基底 Qwen 常見：第一則 generate 出訓練殘留／亂碼，第二則才正常。
    """
    global _llm_chat_primed, model, tokenizer
    if USE_OLLAMA:
        if (
            os.environ.get("LLM_CHAT_PRIME", "1").strip().lower()
            in ("0", "false", "no", "off")
        ):
            _llm_chat_primed = True
            return True
        with _llm_lock:
            if _llm_chat_primed and not force:
                return True
            print("🔥 Ollama 對話探測：先問「你好」並丟棄回覆…")
            ok = ollama_service.prime_chat()
            _llm_chat_primed = True
            return ok

    if model is None or tokenizer is None:
        return False
    if (
        os.environ.get("LLM_CHAT_PRIME", "1").strip().lower()
        in ("0", "false", "no", "off")
    ):
        _llm_chat_primed = True
        return True

    with _llm_lock:
        if _llm_chat_primed and not force:
            return True
        print("🔥 LLM 對話探測：先問「你好」並丟棄回覆（避免使用者看到冷啟動異常）…")
        try:
            msgs = _normalize_chat_messages([
                {
                    "role": "system",
                    "content": (
                        "你是 Semi-Shield Cyber Agent。"
                        "只用一句自然繁體中文打招呼；禁止日文與訓練格式；最後【回答結束】。"
                    ),
                },
                {"role": "user", "content": "你好"},
            ])
            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = "<|user|>\n你好\n<|assistant|>\n"
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=384
            )
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # CPU 探測短一點，避免啟動多等半分鐘
            _prime_n = 16 if not _use_cuda_for_llm() else 48
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=_prime_n,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            gen = out[0][inputs["input_ids"].shape[1] :]
            text = tokenizer.decode(gen, skip_special_tokens=True).strip()
            del out, inputs, gen
            _cuda_empty_cache_quiet()
            _llm_chat_primed = True
            preview = (text[:48] + "…") if len(text) > 48 else (text or "(空)")
            print(f"✅ LLM 對話探測完成（已丟棄）：{preview}")
            return True
        except Exception as e:
            print(f"⚠️ LLM 對話探測失敗：{e}")
            if _is_cuda_fault(e):
                print("⚠️ 對話探測 CUDA 崩潰 → 切 CPU 後備後再探測一次")
                if _failover_llm_to_cpu(str(e)):
                    _llm_chat_primed = False
                    return _prime_llm_chat(force=True)
            # 避免每個請求重試卡死；後續仍靠 sanitize
            _llm_chat_primed = True
            return False


# 啟動時先做對話探測（若暖機已失敗切 CPU，此處對 CPU 模型再探測）
if USE_OLLAMA or (model is not None and tokenizer is not None):
    try:
        _prime_llm_chat(force=True)
    except Exception as _pe:
        print(f"⚠️ 啟動對話探測略過：{_pe}")

# 將啟動時載入的模型放入快取，之後切換可直接复用
if not USE_OLLAMA and model is not None and tokenizer is not None:
    try:
        _boot_key = _normalize_llm_ref(MODEL_PATH)
        _boot_hub = _early_hub_id_from_path(str(MODEL_PATH))
        if _boot_hub:
            _boot_slug = f"base:{_boot_hub}"
            _boot_stage = "base"
            _boot_display = _boot_hub
        else:
            _boot_slug = Path(str(MODEL_PATH)).name
            _boot_stage = "finetuned"
            _boot_display = MODEL_PATH
        _llm_cache_put(
            _boot_key,
            model,
            tokenizer,
            display_path=_boot_display,
            slug=_boot_slug,
            stage=_boot_stage,
        )
        print(f"📦 LLM 快取已啟用（上限 {_LLM_CACHE_MAX}）：{_boot_slug}")
    except Exception as _ce:
        print(f"⚠️ LLM 快取註冊略過：{_ce}")

# 結束標記：生成到這裡就停（含日文誤用結束語，避免後面又重寫一份）
_STOP_STRINGS = (
    "【回答結束】",
    "【報告結束】",
    "答え終了",
    "回答終了",
    "答え終わり",
)
_stop_ids_list = []
if tokenizer is not None:
    for s in _STOP_STRINGS:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            _stop_ids_list.append(ids)

# =======================================================
# LLM 推理核心函數
# =======================================================
def _looks_truncated(reply):
    """粗略判斷回答是否被 max_new_tokens 截斷。"""
    if not reply:
        return True
    text = reply.strip().replace("\ufffd", "").rstrip()
    if not text:
        return True
    if "【報告結束】" in text or "【回答結束】" in text:
        return False
    if "\ufffd" in (reply or ""):
        return True
    # 明顯寫到一半：結尾不是標點，或停在常見半截詞
    if text.endswith((
        "審", "建", "建置", "建置完", "建議", "防護", "不符", "摘要", "分析",
        "：", ":", "-", "、", "字段", "欄位", "以便", "加入", "回", "並", "及",
        "與", "的", "了", "在", "能", "會", "可", "並能", "並且",
    )):
        return True
    # 英文半截詞（如 chartjs、type、field）
    if text[-1].isalnum() and not text.endswith(("pass", "fail", "review", "OK")):
        # 最後一段若沒有中文句號且很短收尾，視為截斷
        tail = text[-40:]
        if "。" not in tail and "！" not in tail and "？" not in tail:
            return True
    if text[-1] not in "。！？.!?」』）)】`":
        return True
    # 三段式報告缺段
    has_s1 = "摘要" in text
    has_s2 = "不符" in text or "分析" in text
    has_s3 = "建議" in text or "修補" in text
    if has_s1 and has_s2 and not has_s3:
        return True
    return False


def _is_phi_model_active() -> bool:
    """目前作用中模型是否為 Phi 系列（較易條列循環／照抄指示）。"""
    blob = str(MODEL_PATH or "").lower()
    try:
        blob += " " + str((_current_llm_info() or {}).get("slug") or "").lower()
    except Exception:
        pass
    return "phi" in blob


def _is_small_or_gemma_model() -> bool:
    """Gemma／小参數模型：需更短 prompt，閒聊易幻覺。"""
    if USE_OLLAMA:
        return ollama_service.is_small_model(MODEL_PATH)
    blob = str(MODEL_PATH or "").lower()
    try:
        blob += " " + str((_current_llm_info() or {}).get("slug") or "").lower()
        blob += " " + str((_current_llm_info() or {}).get("path") or "").lower()
    except Exception:
        pass
    return any(
        k in blob
        for k in ("gemma", "0.5b", "1.5b", "2b", "phi-4-mini", "phi4-mini", "phi3")
    )


def _strip_draft_preamble(text: str) -> str:
    """去掉「以下是根據底稿改寫的回覆」等元開場白。"""
    if not text:
        return text
    t = text.strip()
    t = re.sub(
        r"^(?:好的[，,]?)?(?:以下是|以下為|這是)?"
        r"(?:根據(?:您|你)提供的)?"
        r".{0,96}?(?:改寫|重写)的?(?:回覆|回复|回答)[:：]\s*",
        "",
        t,
        count=1,
        flags=re.I,
    )
    lines = [ln for ln in t.splitlines()]
    while lines:
        first = lines[0].strip()
        if not first:
            lines.pop(0)
            continue
        if re.fullmatch(r"的?(?:回覆|回复|回答)[:：]?", first):
            lines.pop(0)
            continue
        if (
            len(first) <= 120
            and re.search(
                r"(?:底稿|草稿|結構化).*?(?:改寫|重写)|"
                r"(?:以下是|以下為).*?(?:底稿|草稿|改寫|重写)",
                first,
                re.I,
            )
            and not re.search(r"A\.\d+\.\d+|status\s*=", first, re.I)
        ):
            lines.pop(0)
            continue
        break
    return "\n".join(lines).strip() or text.strip()


def _polish_chat_output(text: str) -> str:
    """修正聊天常見格式殘留（孤立【、重複句號、model 殘留等）。"""
    if not text:
        return text
    t = _strip_draft_preamble(text)
    t = re.sub(r"【(?:回答|報告)結束】", "", t)
    t = re.sub(r"【(?![^【\n]{0,80}】)", "", t)  # 未閉合的【
    t = re.sub(r"【\s*$", "", t)
    t = re.sub(r"。{2,}", "。", t)
    t = re.sub(r"\s+【\s*$", "", t)
    kept = []
    for ln in t.splitlines():
        s = ln.strip()
        if re.fullmatch(r"(?i)model|assistant|user|system|human|ai", s):
            continue
        if re.fullmatch(r"(?i)(?:the\s+)?model\s*(?:output|response)?", s):
            continue
        kept.append(ln)
    t = "\n".join(kept)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def _looks_like_offtopic_hallucination(text: str) -> bool:
    """離題問題卻捏造天氣／地名／工廠場景。"""
    if not text:
        return False
    return bool(
        re.search(
            r"台東|外電廠|發電廠.*工人|溫度.*舒適|風和日麗|攝氏\s*\d+度|"
            r"今天.*(晴天|下雨|陣雨)|氣溫.*左右",
            text,
        )
    )


def _wants_security_concept(user_message: str) -> bool:
    t = (user_message or "").strip()
    if not t:
        return False
    return bool(
        re.search(
            r"紅隊|藍隊|紫隊|滲透測試|渗透测试|penetration|red\s*team|blue\s*team|"
            r"威脅獵捕|threat\s*hunt|社交工程|social\s*engineering|"
            r"資安演練|攻防演練|tabletop",
            t,
            re.I,
        )
    )


def build_security_concept_knowledge(user_message: str = "") -> str:
    """資安概念題 grounded 事實卡（避免小模型過簡或錯誤結論）。"""
    t = user_message or ""
    if re.search(r"紅隊", t, re.I):
        return (
            "【正確事實｜紅隊演練】\n"
            "紅隊演練是以攻擊者視角，在**授權範圍與 ROE** 內模擬真實 TTP，"
            "檢驗偵測、應變、修補與控制措施是否有效。\n"
            "常見項目：偵察、釣魚／社交工程、漏洞利用、權限提升、橫向移動、資料外洩演練；"
            "需留存 evidence 供複盤。\n"
            "與 ISO/IEC 27001:2022 可對照：A.5.24–26 事件管理、A.8.8 弱點管理、"
            "A.8.16 監控活動、持續改善。\n"
            "【禁止】「完全沒漏洞就不用再投資資安」— 應依風險分級排修、驗證改善並持續監控。"
        )
    if re.search(r"藍隊|威脅獵捕", t, re.I):
        return (
            "【正確事實｜藍隊】\n"
            "藍隊負責日常監控、告警 triage、威脅獵捕、事件調查與 hardening；"
            "與紅隊演練搭配可驗證 SOC 流程與控制是否有效。"
        )
    return (
        "【正確事實｜資安演練】\n"
        "受控演練用於驗證人員、流程與技術控制；"
        "須有授權、範圍、證據留存與改善追蹤，不可當未授權攻擊。"
    )


def wants_security_concept_explain(user_message: str) -> bool:
    """純資安概念問答（什麼是紅隊／你知道…嗎）→ 直接 grounded，不走小模型。"""
    if not _wants_security_concept(user_message):
        return False
    t = (user_message or "").strip()
    if wants_visual(t):
        return False
    # 延伸／情境／實務題 → LLM + grounded fallback
    if re.search(
        r"注意|如何|怎麼|為什麼|風險|步驟|實務|範例|"
        r"差異|比較|vs|對照|在.*環境|OT|工控|半導體",
        t,
        re.I,
    ):
        return False
    if re.search(
        r"什麼是|是什麼|是甚麼|何謂|怎麼定義|如何定義|"
        r"你知道.*嗎|請介紹|請解釋|請說明|介绍一下|介紹一下",
        t,
        re.I,
    ):
        return True
    if len(t) <= 10 and re.fullmatch(r"紅隊|藍隊|紫隊|滲透測試?", t, re.I):
        return True
    return False


def _security_concept_reply_is_bad(text: str) -> bool:
    """資安概念回答品質過低或含已知錯誤結論。"""
    r = (text or "").strip()
    if not r:
        return True
    if _is_trusted_grounded_security_reply(r):
        return False
    if _cjk_count(r) < 55:
        return True
    if re.search(r"【(?![^【\n]{0,120}】)|【\s*$", r):
        return True
    if _has_bad_security_conclusion(r):
        return True
    if re.search(r"回答結束", r) and _cjk_count(r) < 90:
        return True
    if _looks_like_train_leak(r) or _looks_like_offtopic_hallucination(r):
        return True
    return False


def _has_bad_security_conclusion(text: str) -> bool:
    """偵測錯誤肯定句（零漏洞不用資安），排除「不能解讀為…」等否定引用。"""
    for m in re.finditer(
        r"不必.{0,8}資安|不用.{0,8}投資.{0,6}資安|"
        r"完全沒.{0,6}漏洞|沒有漏洞.{0,6}不用|零漏洞.{0,6}不用",
        text or "",
    ):
        start = m.start()
        ctx = (text or "")[max(0, start - 16): start]
        if re.search(r"不能|不可|勿|並非|不是|禁止|解讀為[「\"']", ctx):
            continue
        return True
    return False


def _is_trusted_grounded_security_reply(text: str) -> bool:
    """已 curated 的 grounded 資安概念回答，勿再觸發品質 fallback。"""
    r = (text or "").strip()
    return bool(
        re.search(
            r"^\*\*(?:紅隊演練|藍隊|紫隊|滲透測試|威脅獵捕|社交工程|資安演練)",
            r,
        )
    )


def build_security_concept_reply(user_message: str = "") -> str:
    """資安概念題：高品質 grounded 回答（供直接回覆或 LLM 失敗 fallback）。"""
    t = user_message or ""
    if re.search(r"紅隊", t, re.I):
        return (
            "**紅隊演練（Red Team Exercise）**是在組織**書面授權**與 **ROE（交戰規則）** "
            "範圍內，由攻擊方模擬真實對手 **TTP**，檢驗偵測、應變、修補與控制措施是否有效。\n\n"
            "**與滲透測試的差異**：滲透測試常聚焦特定系統或漏洞驗證；"
            "紅隊演練更偏**全流程、多階段**（含社交工程、橫向移動、資料外洩演練），"
            "目標是驗證整體防禦鏈而非單點。\n\n"
            "**典型階段**：偵察 → 初始入侵 → 權限提升 → 橫向移動 → 達成目標 → "
            "報告與複盤；過程須留存 **evidence**（時間軸、IOC、截圖／log）供改善追蹤。\n\n"
            "**OT／半導體情境注意**：須區分 IT 與 OT 網段；"
            "不可對運行中 **SIS／PLC** 做破壞性 exploit；"
            "演練前需變更單、備援與回滾計畫，並遵守 IEC 62443 分區原則。\n\n"
            "**ISO/IEC 27001:2022 對照**：A.5.24–26 事件管理、A.8.8 弱點管理、"
            "A.8.16 監控活動；產出應納入持續改善與稽核證據鏈。\n\n"
            "補充：即使某次掃描未發現漏洞，仍須依**風險分級**持續監控、修補與驗證改善，"
            "不能解讀為「不必再投資資安」。"
        )
    if re.search(r"藍隊", t, re.I):
        return (
            "**藍隊（Blue Team）**負責日常資安防禦與事件應變，"
            "包含 SOC 監控、告警 triage、威脅獵捕（Threat Hunting）、"
            "事件調查、日誌分析與系統 **hardening**。\n\n"
            "與**紅隊演練**搭配時，藍隊要在演練期間偵測、阻斷並調查紅隊行為，"
            "事後共同複盤：哪些控制有效、哪些告警被忽略、流程哪裡要補強。\n\n"
            "**ISO/IEC 27001 對照**：A.5.24–26 事件管理、A.8.16 監控活動、"
            "A.8.23 資訊安全事件管理流程。\n\n"
            "本平台 Semi-Shield 可協助將 OT syslog 對映控制項，"
            "作為藍隊日常監控與稽核 evidence 的輔助。"
        )
    if re.search(r"紫隊|purple", t, re.I):
        return (
            "**紫隊（Purple Team）**是紅隊與藍隊的**協作模式**："
            "紅隊負責模擬攻擊，藍隊同步驗證偵測與應變，"
            "雙方即時交流 TTP 與防禦缺口，縮短「攻擊成功 ↔ 被發現」的落差。\n\n"
            "相較傳統紅藍對抗，紫隊更強調**知識轉移**與**控制驗證**，"
            "適合在導入新 SIEM 規則或 OT 監控策略後快速驗證成效。"
        )
    if re.search(r"滲透|penetration", t, re.I):
        return (
            "**滲透測試（Penetration Test）**是在授權範圍內，"
            "對指定系統／應用／網段進行有計畫的漏洞驗證與利用測試，"
            "產出風險報告與修補建議。\n\n"
            "**與紅隊演練差異**：滲透測試通常有明確 scope 與時間盒；"
            "紅隊演練更模擬 APT 多階段、多向量，並檢驗 SOC 偵測與應變。\n\n"
            "**OT 注意**：須在測試床或維護窗口執行；"
            "禁止對生產 PLC／SIS 做未授權 exploit。"
        )
    if re.search(r"威脅獵捕|threat\s*hunt", t, re.I):
        return (
            "**威脅獵捕（Threat Hunting）**是藍隊主動、假設驅動地搜尋潛藏威脅，"
            "而非只等 SIEM 告警。常見做法：建立假設 → 查詢 log／EDR → "
            "驗證 IOC → 事件化處理。\n\n"
            "在 OT 環境可結合 Cisco syslog、存取異常、組態變更（A.8.19）"
            "與惡意事件（A.8.7）做關聯分析。"
        )
    if re.search(r"社交工程|social\s*engineering", t, re.I):
        return (
            "**社交工程（Social Engineering）**是透過釣魚信、假冒身份、"
            "電話詐騙等方式誘使使用者洩漏憑證或執行惡意操作。\n\n"
            "紅隊演練常含釣魚模擬；防禦需搭配 A.6 人員安全意識、"
            "A.8.5 身分驗證、A.8.7 惡意程式防護與使用者通報流程。"
        )
    return (
        "**資安演練**是在**書面授權**下，模擬攻擊或事件情境，"
        "驗證人員、流程與技術控制是否有效。\n\n"
        "常見類型：紅隊（攻擊模擬）、藍隊（防禦應變）、紫隊（協作驗證）、"
        "Tabletop（桌面推演）。\n\n"
        "無論結果如何，都須留存 evidence、開立改善項並追蹤至關閉；"
        "不可將演練當作未授權攻擊，也不可因「暫時沒漏洞」而停止資安投資。"
    )


def _chat_grounded_fallback_enabled() -> bool:
    """固定 canned 回答；預設關閉，知識題一律 LLM（設 CHAT_GROUNDED_FALLBACK=1 才啟用）。"""
    return os.environ.get("CHAT_GROUNDED_FALLBACK", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _finalize_chat_reply(reply: str, user_message: str = "") -> str:
    """聊天回覆最後一道：格式修正；固定回答僅在 CHAT_GROUNDED_FALLBACK=1 時。"""
    r = _polish_chat_output(reply or "")
    if not _chat_grounded_fallback_enabled():
        return r
    if _is_trusted_grounded_security_reply(r):
        return r
    if _is_trusted_grounded_iso27001_reply(r):
        return r
    if _wants_security_concept(user_message) and _security_concept_reply_is_bad(r):
        print("⚠️ 資安概念回答品質不足，改用 grounded（CHAT_GROUNDED_FALLBACK=1）")
        return build_security_concept_reply(user_message)
    if _wants_iso27001_topic(user_message) and _iso27001_reply_is_bad(r):
        print("⚠️ ISO 27001 回答品質不足，改用 grounded（CHAT_GROUNDED_FALLBACK=1）")
        return build_iso27001_overview_reply(user_message)
    return r


def should_use_grounded_casual_reply(user_message: str) -> bool:
    """
    離題／寒暄固定短答；預設關閉，一律走 LLM。
    設 CHAT_GROUNDED_CASUAL=1 才對天氣等離題用固定短答。
    """
    if os.environ.get("CHAT_GROUNDED_CASUAL", "0").strip().lower() not in (
        "1", "true", "yes", "on",
    ):
        return False
    if ENABLE_CASUAL_FIXED_REPLY:
        return is_casual_chat(user_message) or is_off_topic_chat(user_message)
    if is_off_topic_chat(user_message):
        return True
    # 「你好啊今天天氣如何」：is_casual 為真但 is_off_topic 原先為假
    if is_casual_chat(user_message) and re.search(
        r"天氣|氣溫|下雨|幾號|幾點|星期|日期",
        user_message or "",
        re.I,
    ):
        return True
    return False


def _strip_instruction_echo(text: str) -> str:
    """去掉把系統指示複述進回答的句子。"""
    if not text:
        return text
    keep = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            keep.append(ln)
            continue
        if re.match(
            r"^請(?:用|只用)?(?:繁體)?中文回答",
            s,
        ):
            continue
        if re.search(r"使用\s*Markdown\s*格式", s) and len(s) <= 48:
            continue
        if re.match(r"^請以繁體中文", s) and len(s) <= 40:
            continue
        if re.match(r"^(?:system|assistant|user)\s*[:：]?\s*$", s, re.I):
            continue
        if re.search(r"語言硬性規定|防幻覺鐵律|總字數\s*≤", s):
            continue
        keep.append(ln)
    return "\n".join(keep).strip()


def _collapse_juti_bullet_loop(text: str) -> str:
    """折掉「- 具體要求／具體情況…」這類循環條列（Phi-4 常見）。"""
    if not text or text.count("具體") < 6:
        return text
    lines = text.splitlines()
    out = []
    seen = []
    cycle_closed = False
    juti_total = 0
    for ln in lines:
        m = re.match(r"^([-*•]\s*)?(具體[\u4e00-\u9fffA-Za-z0-9]{1,12})\s*$", ln.strip())
        if not m:
            if cycle_closed and re.match(r"^[-*•]\s*具體", ln.strip()):
                continue
            out.append(ln)
            continue
        key = m.group(2)
        juti_total += 1
        if cycle_closed:
            continue
        if key in seen:
            # 第二輪開始 → 截斷後續具體條列
            cycle_closed = True
            continue
        seen.append(key)
        out.append(f"- {key}")
        if len(seen) >= 10:
            cycle_closed = True
    # 幾乎全是具體條列且無完整句子 → 視為無效循環
    body = "\n".join(out).strip()
    if juti_total >= 8 and _cjk_count(re.sub(r"具體\S+", "", body)) < 24:
        return ""
    return body


def _looks_like_list_loop(text: str) -> bool:
    """是否像條列詞彙循環、幾乎沒有真正回答。"""
    if not text:
        return True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    juti_keys = []
    for ln in lines:
        m = re.match(r"^[-*•]?\s*(具體[\u4e00-\u9fffA-Za-z0-9]{1,12})\s*$", ln)
        if m:
            juti_keys.append(m.group(1))
    if len(juti_keys) >= 6:
        uniq = set(juti_keys)
        # 同一批詞出現兩輪以上，或幾乎沒有非「具體」正文
        if len(juti_keys) >= max(6, len(uniq) * 2):
            return True
        if _cjk_count(re.sub(r"具體\S+", "", text)) < 40:
            return True
    if text.count("具體") >= 12 and _cjk_count(re.sub(r"具體\S+", "", text)) < 40:
        return True
    for key in set(juti_keys):
        if juti_keys.count(key) >= 3:
            return True
    return False


def _collapse_repetitions(text: str) -> str:
    """去除模型陷入的重複句 / 重複片段。"""
    if not text:
        return text

    text = _strip_instruction_echo(text)

    # 1) 連續重複同一句（含「請您在 bar chart...」這類 loop）
    # 以句號/感嘆/問號切段後去重保序（勿依 \\n 切段，否則編號清單會被併成一行）
    parts = re.split(r'(?<=[。！？])', text)
    cleaned, prev, streak = [], None, 0
    for p in parts:
        norm = re.sub(r'\s+', '', p)
        if not norm:
            continue
        if norm == prev:
            streak += 1
            if streak >= 1:  # 同一句只留一次
                continue
        else:
            streak = 0
            prev = norm
            cleaned.append(p)
    text = "".join(cleaned).strip()

    # 2) 滑動偵測：同一子字串連續出現 >=3 次，砍掉後面
    for window in (12, 16, 24, 32, 48, 64):
        if len(text) < window * 3:
            continue
        # 找最長重複前綴尾
        for i in range(0, len(text) - window * 3):
            chunk = text[i:i + window]
            if not chunk.strip():
                continue
            triple = chunk * 3
            pos = text.find(triple)
            if pos >= 0:
                # 保留到第一次 chunk 結束
                text = text[: pos + window].rstrip()
                break

    # 3) 常見 loop 關鍵片語：從第二次起截斷
    loop_markers = [
        "請您在 bar chart 代碼中加入",
        "請您在 bar chart 代碼中加入 'type'",
        "以便 chartjs-chart-bar",
        "請在 bar chart",
        "具體要求",
        "具體情況",
    ]
    for marker in loop_markers:
        first = text.find(marker)
        if first < 0:
            continue
        second = text.find(marker, first + len(marker))
        if second >= 0 and marker.startswith("具體"):
            # 具體* 允許第一次列表內各出現一次；第三次同詞才砍
            third = text.find(marker, second + len(marker))
            if third >= 0:
                text = text[:third].rstrip()
        elif second >= 0 and not marker.startswith("具體"):
            text = text[:second].rstrip()

    # 4) 「- 具體XXX」整輪循環
    text = _collapse_juti_bullet_loop(text)
    return text.strip()


def _to_traditional(text: str) -> str:
    """簡體 → 繁體（zh-TW）。"""
    if not text:
        return text
    if zhconv is None:
        return text
    try:
        return zhconv.convert(text, "zh-tw")
    except Exception:
        return text


def _has_japanese(text: str) -> bool:
    if not text:
        return False
    return bool(_RE_HIRAGANA.search(text) or _RE_KATAKANA.search(text) or _RE_JP_MARKERS.search(text))


def _japanese_score(text: str) -> int:
    if not text:
        return 0
    return (
        len(_RE_HIRAGANA.findall(text))
        + len(_RE_KATAKANA.findall(text))
        + (3 if _RE_JP_MARKERS.search(text) else 0)
    )


def _cjk_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text or ""))


def _looks_like_train_leak(text: str) -> bool:
    """微調／基底模型殘留：few-shot『使用者：…』、亂碼 token、內容標籤。"""
    t = (text or "").strip()
    if not t:
        return False
    if re.search(
        r"使用者\s*[:：]\s*[「\"']|"
        r"(?:^|\n)\s*(?:Human|Assistant|User|System)\s*[:：]|"
        r"###\s*(?:Instruction|Response|輸入|輸出)|"
        r"請提供一個關於.{0,40}(?:資安|半導體|合規).{0,20}案例|"
        r"\bnudity\b|tearaway|"
        r"[A-Za-z]{3,}(?:\s*&\s*[A-Za-z]{3,}){1,}",
        t,
        re.I,
    ):
        return True
    first = (t.splitlines()[0] or "").strip()
    if first and _cjk_count(first) < 2 and re.search(
        r"[A-Za-z]{2,}[-_&][A-Za-z]{2,}|[A-Za-z0-9_\-]{6,}[@\"'`]",
        first,
    ):
        return True
    # 整段幾乎沒中文、卻像標籤／亂碼
    if len(t) <= 80 and _cjk_count(t) < 4 and re.search(r"[A-Za-z]{4,}", t):
        return True
    return False


def _needs_zh_retry(text: str) -> bool:
    """輸出仍含日文、訓練殘留、或中文過少時，觸發強制重寫。"""
    if not text or not text.strip():
        return True
    if _looks_like_train_leak(text):
        return True
    if _japanese_score(text) >= 3:
        return True
    if _has_japanese(text) and _cjk_count(text) < 40:
        return True
    # 幾乎沒有中文（含短英文亂碼）
    if _cjk_count(text) < 18:
        return True
    if _cjk_count(text) < 24 and len(text) > 40:
        return True
    return False


_PROMPT_LEAK_RE = re.compile(
    r"格式撰寫|撰寫診斷|撰寫報告|必要的資訊和建議|所有必要的資訊|"
    r"合作規診斷|請以[「\"'].*報告|回答第一行必須|總字數必須|"
    r"總字數\s*≤|不要補劇本|勿虛構監控|最後一行【|"
    r"OUTPUT_FORMAT|禁止輸出「?##|只用繁體中文重寫|"
    r"請直接輸出繁體中文\s*Markdown|從「##\s*地端|"
    r"每段至少\s*1\s*句|不可只輸出\s*#|"
    r"【內部(?:監控|知識|日誌)|禁止照抄|轉述重點|"
    r"開始輸入答案|開始輸出答案|開始作答|輸入答案|"
    r"開始輸入|開始輸出|在此輸入|請開始寫|"
    r"一、二、三每段都要|禁止貼上日誌原文|勿虛構攻擊|"
    r"防幻覺鐵律|禁止捏造",
    re.I,
)


def _is_prompt_instruction_line(line: str) -> bool:
    """偵測模型把 system/user 指示句當成報告正文。"""
    s = (line or "").strip()
    if not s:
        return False
    s = re.sub(r"^[-*•]\s+", "", s)
    if _PROMPT_LEAK_RE.search(s):
        return True
    # 「…報告」格式／開始… 這類元指令
    if re.search(r"[「\"'].{0,32}報告[」\"'].{0,16}(?:格式|開始)", s):
        return True
    if re.search(r"智慧合規診斷報告.{0,12}開始", s):
        return True
    if re.search(r"包含所有必要|請依(?:下列|以下)格式|嚴格使用：\s*##", s):
        return True
    # 整句只在複誦標題／指示，沒有實際稽核內容
    if re.fullmatch(
        r"[「\"'【\[]?[#\s]*地端\s*LLM\s*智慧合規診斷報告[」\"'】\]]?.{0,24}",
        s,
    ):
        return True
    if re.search(r"(?:底稿|草稿).*?(?:改寫|重写).*?(?:回覆|回复|回答)", s):
        return True
    return False


def _strip_meta_leak(text: str) -> str:
    """清除提示詞洩漏、日文結束語、字數指示。"""
    if not text:
        return text
    text = re.sub(r"答え終了\**", "", text)
    text = re.sub(r"回答終了\**", "", text)
    text = re.sub(r"[*＊]{2,}", "", text)
    text = re.sub(r"[（(]\s*\d+\s*[~～\-－]\s*\d+\s*(?:單語|単語|字|詞)[^）)]*[）)]", "", text)
    text = re.sub(r"[（(][^）)]*(?:以內|以内|必須|禁止日文|只用繁體)[^）)]*[）)]", "", text)
    # 佔領 OS / 奇怪混雜行
    text = re.sub(r"(?m)^.*(?:佔領|セキュリティ・ド|答え).*$", "", text)
    # 整行提示詞洩漏
    kept = []
    for ln in text.splitlines():
        if _is_prompt_instruction_line(ln):
            continue
        kept.append(ln)
    return "\n".join(kept)


def _looks_like_context_leak(text: str) -> bool:
    """判斷前綴是否像被模型照抄的 RAG／Log／監控原文。"""
    if not text:
        return False
    markers = (
        r"\[知識\d*",
        r"RAG\s*知識",
        r"監控資料",
        r"SNMP\s*記錄",
        r"寫入\s*\d+\s*檔",
        r"核心網監測",
        r"core network monitoring",
        r"/var/log",
        r"WLC-\d+",
        r"WS-C\d+",
        r"raw\.ndjson",
        r"5m\.ndjson",
        r"score\s*=\s*0\.\d+",
        r"已略過大段英文原文",
        r"Jul\s+\d{1,2}\s+\d{2}:",
        r"192\.168\.\d+",
    )
    hit = sum(1 for p in markers if re.search(p, text, flags=re.I))
    return hit >= 1 or (len(text) > 80 and _cjk_count(text) < 20)


def _strip_context_leak(text: str) -> str:
    """
    去掉模型照抄的 RAG／Log 前綴，只保留真正的診斷報告。
    常見：先貼監控原文 → ## AI: → 正文
    """
    if not text:
        return text

    # 若有明確報告起點，且前面像洩漏，直接切掉
    start_pats = [
        r"(?m)^##\s*地端\s*LLM",
        r"(?m)^##\s*一、",
        r"(?m)^##\s*事件經過",
        r"(?m)^#{1,4}\s*AI\s*[:：]\s*",
        r"(?m)^一、事件經過摘要",
    ]
    for pat in start_pats:
        m = re.search(pat, text)
        if not m:
            continue
        head = text[: m.start()]
        if m.start() >= 20 and _looks_like_context_leak(head):
            text = text[m.start():]
            break
        # 即使前面不長，只要含明顯 Log 標記也切
        if re.search(r"SNMP|WLC-|/var/log|寫入\s*\d+\s*檔|\[知識", head, flags=re.I):
            text = text[m.start():]
            break

    # 去掉 "## AI:" 包裝列
    text = re.sub(r"(?m)^#{1,4}\s*AI\s*[:：]\s*", "", text)

    cleaned = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            cleaned.append("")
            continue
        # 原始日誌／設備流水列
        if re.search(
            r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:",
            raw,
        ):
            continue
        if re.search(r"\[知識\d*|score\s*=\s*0\.\d+|寫入\s*\d+\s*檔\s*SNMP", raw, flags=re.I):
            continue
        if re.search(r"\[核心網監測\]|\[core network monitoring\]", raw, flags=re.I):
            continue
        if re.search(r"已略過大段英文原文", raw):
            continue
        if re.search(r"以下是(?:後端|RAG|監控)|知識庫檢索結果|日誌摘要：", raw):
            continue
        cleaned.append(line.rstrip())

    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # 若切完後沒有標題，補上標準報告殼
    if text and not re.search(r"(?m)^##\s+", text):
        text = (
            "## 地端 LLM 智慧合規診斷報告\n\n"
            "## 一、事件經過摘要\n\n"
            f"{text}\n\n"
            "【報告結束】"
        )
    return text


def _prefer_chinese_pass(text: str) -> str:
    """
    若模型先寫日文版再寫繁中版，只保留較像繁中的那一段。
    常見：日文三段 → 答え終了 → 繁中三段。
    """
    if not text:
        return text

    # 以「第二次出現的報告標題 / 一、」作為繁中重寫起點
    restart_pats = [
        r"#{1,4}\s*地端\s*LLM",
        r"#{1,4}\s*一、(?:事件|小報|概要|摘要)",
        r"#{1,4}\s*一、事件經過摘要",
        r"#{1,4}\s*一、小報概述",
    ]
    # 若前半是日文、後半有繁中重寫，切到後半
    for pat in restart_pats:
        matches = list(re.finditer(pat, text, flags=re.I))
        if len(matches) >= 2:
            # 選「日文分數較低」的那次起點
            best_start, best_score = 0, 10**9
            for m in matches:
                chunk = text[m.start():]
                sc = _japanese_score(chunk)
                if sc < best_score:
                    best_score, best_start = sc, m.start()
            text = text[best_start:]
            break
        if len(matches) == 1 and _japanese_score(text[: matches[0].start()]) >= 5:
            text = text[matches[0].start():]
            break

    # 答案終了之後若還有內容，取分數較好的一側
    for sep in ("答え終了", "回答終了", "答え終わり"):
        if sep in text:
            left, right = text.split(sep, 1)
            if _japanese_score(right) < _japanese_score(left) and _cjk_count(right) >= 20:
                text = right
            else:
                text = left
            break
    return text.strip()


def _drop_japanese_lines(text: str) -> str:
    """逐行剔除含假名／日文標記的內容。"""
    if not text:
        return text
    kept = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            kept.append("")
            continue
        if _RE_HIRAGANA.search(raw) or _RE_KATAKANA.search(raw):
            continue
        if _RE_JP_MARKERS.search(raw):
            continue
        # 標題含日文漢字混假名已在上面處理；純日文漢字標題（イベント）也擋
        if re.search(r"イベント|経過概要|危険評価|修正提案|安全対策", raw):
            continue
        kept.append(line.rstrip())
    text = "\n".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _dedupe_repeated_sections(text: str) -> str:
    """同一結構寫兩次（兩個「## 一、」）時只留第一份完整繁中。"""
    if not text:
        return text
    # 找所有「## 一、」
    starts = [m.start() for m in re.finditer(r"(?m)^##\s*一、", text)]
    if len(starts) >= 2:
        # 取日文分數最低的區段
        candidates = []
        for i, st in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(text)
            # 向後擴到下一份「## 地端」或結尾；此區段含二、三
            chunk = text[st:end]
            # 若下一段很短，併到文件尾
            if i == len(starts) - 1:
                chunk = text[st:]
            else:
                # 包含到下一「一、」之前的二、三
                chunk = text[st:starts[i + 1]]
            candidates.append(chunk)
        # 若第一段之後其實是完整第二份，比較整份 half
        half = len(text) // 2
        alt = [text[:half], text[half:]]
        pool = candidates + alt
        best = min(pool, key=lambda c: (_japanese_score(c), -_cjk_count(c), len(c)))
        if _cjk_count(best) >= 20:
            # 若 best 沒有報告標題，補上前綴標題（若原文有）
            hero = re.search(r"(?m)^##\s*地端\s*LLM[^\n]*", text)
            if hero and "## 地端" not in best:
                best = hero.group(0) + "\n\n" + best
            text = best
    return text.strip()


def _strip_garbage(text: str) -> str:
    """清除字母亂碼、控制項幻覺、全大寫英文牆、日文。"""
    if not text:
        return text

    text = _strip_meta_leak(text)
    text = _prefer_chinese_pass(text)
    text = _drop_japanese_lines(text)
    text = _dedupe_repeated_sections(text)

    # 保留 fenced code 區塊（圖表 JSON / 必要時的短 log），其餘段落清理
    parts = re.split(r"(```[\s\S]*?```)", text)
    out = []
    for part in parts:
        if part.startswith("```"):
            # 過長的假 log 區塊：截短並標註
            if len(part) > 500 and not re.search(r"```chart|```mermaid|```json", part, re.I):
                lang = re.match(r"```(\w*)", part)
                tag = (lang.group(1) if lang else "") or "text"
                body = part.strip("`").split("\n", 1)[-1][:280]
                out.append(f"```{tag}\n{body}\n...（日誌過長已省略）\n```")
            else:
                out.append(part)
            continue

        # 括號內 / 裸露的字母點線亂碼
        part = re.sub(
            r"[\(（]\s*[A-Za-z0-9]+(?:[\s]*[\.\-_/]+[\s]*[A-Za-z0-9]+){5,}\s*[\)）]",
            "",
            part,
        )
        part = re.sub(
            r"(?<![\u4e00-\u9fff])[A-Za-z0-9]+(?:[\.\-_/]+[A-Za-z0-9]+){7,}(?![\u4e00-\u9fff])",
            "",
            part,
        )
        part = re.sub(r"(?:[A-Za-z][\.\-_/]+){8,}[A-Za-z]", "", part)
        part = re.sub(
            r"[\(（]\s*A\.\d+\.[A-Za-z][A-Za-z0-9.\-_/]{0,24}\s*[\)）]",
            "",
            part,
        )
        part = re.sub(r"[\)）]{2,}", "）", part)
        part = re.sub(r"[\(（]{2,}", "（", part)
        part = re.sub(r"[\(（]\s*[\)）]", "", part)

        cleaned_lines = []
        for line in part.splitlines():
            raw = line.strip()
            if not raw:
                cleaned_lines.append("")
                continue
            if _has_japanese(raw):
                continue
            letters = re.findall(r"[A-Za-z]", raw)
            cjk = re.findall(r"[\u4e00-\u9fff]", raw)
            if len(raw) >= 80 and len(letters) >= 40 and len(cjk) <= 2:
                if not cleaned_lines or cleaned_lines[-1] != "（已略過大段英文原文，改以繁體中文結論為準。）":
                    cleaned_lines.append("（已略過大段英文原文，改以繁體中文結論為準。）")
                continue
            if len(raw) >= 60 and raw.isupper() and len(cjk) == 0:
                continue
            cleaned_lines.append(line.rstrip())
        out.append("\n".join(cleaned_lines))

    text = "".join(out)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _classify_section_title(title: str):
    """把雜亂標題對應到標準三段；報告總標題回傳 hero。"""
    t = (title or "").strip()
    t = re.sub(r"^#{1,6}\s*", "", t)
    t = re.sub(r"^[*＊]+|[*＊]+$", "", t).strip()
    t = re.sub(r"^[一二三四五123]\s*[、.．]\s*", "", t).strip()
    if not t or t in {"#", "##", "###", "####"}:
        return None
    if re.search(r"智慧合規診斷報告|地端\s*LLM.*報告|專業\s*IT\s*合規診斷報告", t, re.I):
        return "hero"
    if re.search(r"事件經過|摘要|概述|概觀|總述|經歷", t):
        return "s1"
    if re.search(r"不合|風險|危害|分析|評估|不符", t):
        return "s2"
    if re.search(r"建議|修補|防護|改善|修正|當事人", t):
        return "s3"
    return "other"


def _split_inline_section_markers(text: str) -> str:
    """把黏在同一行的『。二、xxx』拆成換行標題，避免整段塞進摘要卡。"""
    if not text:
        return text
    # 句號後緊接二、/三、或 ## 二、
    text = re.sub(
        r"([。！？；\n])\s*(#{1,6}\s*)?([一二三][、.．])",
        r"\1\n## \3",
        text,
    )
    text = re.sub(
        r"([。！？；])\s*(不合規／風險分析|具體修補建議|事件經過摘要)",
        r"\1\n## \2",
        text,
    )
    return text


def _sentence_bucket(sentence: str) -> str:
    """依語意把句子分到三段。"""
    s = sentence or ""
    if re.search(r"建議|應立即|修補|更新|強化|建置|啟用|複核|改善|CAPA", s):
        return "s3"
    if re.search(r"不符|風險|弱點|漏洞|過期|失敗|攻擊|未授權|OOM|中斷|危害|不合", s):
        return "s2"
    return "s1"


_CODE_KEYWORDS = {
    "else", "begin", "end", "def", "class", "rescue", "ensure", "case", "when",
    "then", "elsif", "elif", "if", "unless", "while", "until", "for", "do",
    "return", "yield", "super", "self", "nil", "true", "false", "null", "none",
    "try", "except", "finally", "catch", "throw", "new", "var", "let", "const",
    "function", "import", "from", "export", "module", "require", "include",
    "puts", "print", "printf", "console", "lambda", "pass", "break", "continue",
}


def _is_garbage_content_line(line: str) -> bool:
    """過濾程式碼、JSON、提示詞洩漏、無關英文問答。"""
    s = (line or "").strip()
    if not s:
        return True
    s = re.sub(r"^[-*•]\s+", "", s).strip()
    if s in {"#", "##", "###", "---", "***", "```", "{", "}", "[", "]", "/", "\\", "```ruby", "```json", "```python"}:
        return True
    # 純程式關鍵字（else / begin / end …）
    if s.lower() in _CODE_KEYWORDS:
        return True
    if re.fullmatch(r"[A-Za-z_]{1,16}", s) and s.lower() in _CODE_KEYWORDS:
        return True
    # 幾乎只有英文關鍵字、無中文（如 "else begin"）
    if not re.search(r"[\u4e00-\u9fff]", s):
        tokens = re.findall(r"[A-Za-z_]+", s.lower())
        if tokens and all(t in _CODE_KEYWORDS or len(t) <= 2 for t in tokens):
            return True
        if len(s) <= 24 and re.fullmatch(r"[A-Za-z0-9_\s\.=>:\-`]+", s):
            if not re.search(r"\b(ISO|SNMP|RADIUS|SYSLOG|CAPA|OT)\b", s, re.I):
                return True
    if re.fullmatch(r"`{3,}[a-zA-Z0-9_-]*", s):
        return True
    if re.fullmatch(r"[-*_/\\]{1,6}", s):
        return True
    if re.fullmatch(r"[{}\[\],;]+", s):
        return True
    # JSON 碎片
    if re.match(r'^["\']?(control_id|title|description|status|label)["\']?\s*:', s, re.I):
        return True
    if re.search(r"\[REPORT\s*END\]|REPORT\s*END", s, re.I):
        return True
    # 含明顯程式教學／API 洩漏（避免誤殺含 ISO 的正常句）
    if re.search(
        r"\b(respond_to\?|is_string|CoffeeScript|Object\.respond_to|String\s*===|"
        r"my_method|MyClass|string_object|please let me know|"
        r"further assistance|Endpoint Security|ISO 27003|"
        r"p\s+e\.message|e\.message|def\s+\w+|class\s+[A-Z])\b",
        s,
        re.I,
    ):
        return True
    if re.search(r"rescue\s*=>|=>\s*e\b|`[^`]{0,40}`", s):
        return True
    # 整句幾乎沒中文且含多個程式關鍵字
    if _cjk_count(s) < 4:
        hits = sum(1 for t in re.findall(r"[A-Za-z_]+", s.lower()) if t in _CODE_KEYWORDS)
        if hits >= 1 and len(s) <= 40:
            return True
    if re.search(r"我想知道如何使用|檢查一個對象是否|完整代碼片段|用於驗證你的解決方案", s):
        return True
    if re.search(r"以下是一組例子|以下是示例|完整代碼|代碼片段", s):
        return True
    # 提示詞／寫作指示被當成正文（如「格式撰寫診斷報告…」）
    if _is_prompt_instruction_line(s):
        return True
    # 「/ 風險評估」這類殘片標題
    if re.match(r"^[/／]\s*\S{0,12}$", s):
        return True
    # 幾乎全是程式符號
    codeish = len(re.findall(r"[{}()\[\]=<>;]|::|->|=>", s))
    cjk = len(re.findall(r"[\u4e00-\u9fff]", s))
    if codeish >= 2 and cjk <= 4:
        return True
    # 英文牆且無合規關鍵字
    letters = len(re.findall(r"[A-Za-z]", s))
    if len(s) >= 40 and letters >= 28 and cjk <= 4:
        if not re.search(r"ISO|SNMP|RADIUS|SYSLOG|compliance|malware|patch", s, re.I):
            return True
    return False


def _sanitize_report_sentence(text: str) -> str:
    """清掉句中殘留程式碼、結束標記、怪異前綴。"""
    if not text:
        return text
    s = text
    s = re.sub(r"【(?:回答|報告)結束】", "", s)
    s = re.sub(r"\[REPORT\s*END\]", "", s, flags=re.I)
    s = re.sub(r"`[^`]*`", "", s)  # 行內 code
    s = re.sub(r"rescue\s*=>\s*e\b.*", "", s, flags=re.I)
    s = re.sub(r"\bp\s+e\.message\b", "", s, flags=re.I)
    # 句尾／獨立程式關鍵字殘片
    kw = "|".join(re.escape(k) for k in sorted(_CODE_KEYWORDS, key=len, reverse=True))
    s = re.sub(rf"(?:^|[\s，,、;；。])(?:{kw})(?=$|[\s，,、;；。])", " ", s, flags=re.I)
    s = re.sub(r"^[/／]\s*", "", s)
    s = re.sub(r"LLM\s*[^\s]{0,8}合規狀況簡報\s*", "", s)
    s = re.sub(r"LLM\s*[^\s]{0,8}合[綜評綜]\s*", "", s)
    s = re.sub(r"知巧合綜評|晴智合規狀況簡報", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" ，,、;；")
    return s.strip()


def _strip_code_fences(text: str) -> str:
    """整段刪除 fenced code（避免被拆成一堆條列卡）。"""
    if not text:
        return text
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"(?m)^\s*`{3,}.*$", "", text)
    return text


def _build_report_cards(s1, s2, s3) -> str:
    def _clean_lines(lines_list):
        cleaned = []
        for ln in lines_list or []:
            ln = re.sub(r"\s*#{1,6}\s*", " ", str(ln)).strip()
            ln = re.sub(
                r"[一二三]、\s*(?:不合規／風險分析|具體修補建議|事件經過摘要)\s*$",
                "",
                ln,
            ).strip()
            ln = re.sub(r"^[-*•]\s+", "", ln)
            ln = re.sub(r"^\d+\.\s+", "", ln)
            ln = _sanitize_report_sentence(ln)
            if _is_garbage_content_line(ln):
                continue
            if "本段暫無足夠資料" in ln or "請補充日誌後重試" in ln:
                continue
            if "暫時處於等待狀態" in ln:
                continue
            if not ln or ln in {"#", "-", "—"}:
                continue
            if _cjk_count(ln) < 6 and not re.search(r"ISO|A\.\d+", ln):
                continue
            cleaned.append(ln)
        return cleaned

    def _body_unified(lines_list, fallback):
        """三段：導讀＋較完整條列（供 grounded 詳盡診斷）。"""
        cleaned = _clean_lines(lines_list)
        if not cleaned:
            return fallback
        lead = cleaned[0]
        if len(lead) > 360:
            lead = lead[:360] + "…"
        rest = cleaned[1:14]  # 六控制項摘要＋風險／修補條列需留足空間
        if not rest:
            return lead
        bullets = "\n".join(
            f"- {(ln[:280] + '…') if len(ln) > 280 else ln}" for ln in rest
        )
        return f"{lead}\n{bullets}"

    return (
        "## 地端 LLM 智慧合規診斷報告\n\n"
        "## 一、事件經過摘要\n\n"
        f"{_body_unified(s1, '依目前監控資料，尚未觀察到需立即升級的重大事件；建議持續觀察。')}\n\n"
        "## 二、不合規／風險分析\n\n"
        f"{_body_unified(s2, '目前無明確高風險不合規證據；請交叉比對控制項量化指標。')}\n\n"
        "## 三、具體修補建議\n\n"
        f"{_body_unified(s3, '維持現有監控與定期複核；若出現異常日誌再啟動 CAPA。')}\n\n"
        "【報告結束】"
    )


def build_factual_report(control_title=None, metric_summary=None, log_text=None, rag_context=None) -> str:
    """當模型輸出太差時，用控制項／指標／日誌重點組出可讀三卡報告。"""
    title = control_title or "ISO 27001 控制項"
    metric = (metric_summary or "").strip()
    log = (log_text or "").strip()
    if is_empty_log(log):
        log = ""
    # 取日誌前幾行重點（略過提示詞殘片）
    log_lines = []
    for ln in log.splitlines():
        ln = ln.strip()
        if not ln or _is_prompt_instruction_line(ln) or _is_garbage_content_line(ln):
            continue
        if len(ln) > 120:
            ln = ln[:120] + "…"
        log_lines.append(ln)
        if len(log_lines) >= 4:
            break

    s1 = [f"針對控制項「{title}」完成合規檢視（資料來源：設備 TXT 日誌）。"]
    if metric:
        s1.append(f"量化摘要：{metric[:180]}")
    if log_lines:
        s1.append("關鍵日誌摘錄：" + "；".join(log_lines[:2]))
    else:
        s1.append("目前無詳細事件明細，改以量化摘要與知識庫要點評估。")

    s2 = []
    joined = f"{metric} {' '.join(log_lines)} {rag_context or ''}"
    if re.search(r"LOGIN_FAILED|denied|拒絕|未授權|breach|fail\b", joined, re.I):
        s2.append("- 日誌出現登入失敗／拒絕存取特徵，可能涉及身分驗證或授權風險，建議列為 review/fail。")
    elif re.search(r"CONFIG_I|Configured from|組態", joined, re.I):
        s2.append("- 發現組態變更紀錄，需確認變更是否經核准與留存變更單，否則構成變更管理缺口。")
    elif re.search(r"UPDOWN|ILPOWER|insufficient power|link.*(down|up)", joined, re.I):
        s2.append(
            "- 出現介面 UPDOWN／抖動特徵：屬 link flap／連線可用性問題"
            "（線材、對端、PoE 或協商），不是組態檔新增刪除，也不是控制器改配方。"
        )
    elif re.search(r"SEC_LOGIN|LOGIN_SUCCESS|LOGOUT|SSH|tty", joined, re.I):
        s2.append("- 有遠端／本機登入活動；雖未必違規，仍須核對帳號合法性、來源 IP 與 MFA／堡壘機制。")
    elif re.search(r"fail|失敗|過期|expire|OOM|panic|review|警告|warn|timeout|延遲", joined, re.I):
        s2.append("- 日誌或指標出現異常／警示特徵，尚未構成明確重大事故，但需交叉複核佐證。")
    else:
        s2.append("- 目前未見明確攻擊或重大違規證據；風險主要來自監控覆蓋與佐證完整性。")
    if rag_context:
        s2.append("- 已參考知識庫要點，請以實際控制項量測與 TXT 日誌為準。")

    s3 = []
    if re.search(r"LOGIN|AUTH|SSH|SEC_LOGIN", joined, re.I):
        s3.append("- 盤點特權帳號與來源 IP，必要時啟用中央認證（RADIUS/TACACS）與登入告警。")
    if re.search(r"CONFIG", joined, re.I):
        s3.append("- 將組態變更納入審批與備份流程，並保留 before/after 差異供稽核。")
    if re.search(r"UPDOWN|ILPOWER|POWER", joined, re.I):
        s3.append("- 檢查交換器供電／線材與 PoE 預算，並對抖動埠位設定監控門檻。")
    s3.extend([
        "- 保留本次 TXT 掃描證據（指標＋代表性日誌），納入定期合規複核。",
        "- 若異常再現，依控制項啟動矯正與預防措施（CAPA）並追蹤結案。",
    ])
    return _build_report_cards(s1, s2, s3[:4])


def _normalize_report_structure(text: str) -> str:
    """強制整理成標準三卡，並盡量保住可用正文。"""
    if not text:
        return text

    text = _strip_code_fences(text)
    text = _split_inline_section_markers(text)
    # 去掉整行只有 #，以及句尾殘留 #
    text = re.sub(r"(?m)^\s*#{1,6}\s*$", "", text)
    text = re.sub(r"[ \t]*#{1,6}[ \t]*$", "", text, flags=re.M)
    text = re.sub(r"\[REPORT\s*END\]", "", text, flags=re.I)

    lines = text.splitlines()
    buckets = {"s1": [], "s2": [], "s3": [], "other": []}
    current = "other"

    def _push(bucket_key, content):
        content = _sanitize_report_sentence(content)
        if not content or _is_garbage_content_line(content):
            return
        if "本段暫無足夠資料" in content or "暫時處於等待狀態" in content:
            return
        buckets[bucket_key if bucket_key in ("s1", "s2", "s3") else "other"].append(content)

    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        if re.fullmatch(r"【(?:回答|報告)結束】", stripped):
            continue

        # 一行內可能還黏著多個段名：再切一次
        pieces = re.split(r"(?=#{1,6}\s*[一二三][、.．]|^[一二三][、.．])", stripped)
        if len(pieces) > 1:
            for piece in pieces:
                piece = piece.strip()
                if not piece:
                    continue
                if re.match(r"^#{0,6}\s*[一二三][、.．]", piece) or re.match(r"^[一二三][、.．]", piece):
                    title = re.sub(r"^#{1,6}\s*", "", piece).strip()
                    m = re.match(r"^([一二三][、.．][^\n]{0,40}?)(?:\s+|$)(.*)$", title)
                    if m and m.group(2):
                        kind = _classify_section_title(m.group(1))
                        if kind in ("s1", "s2", "s3"):
                            current = kind
                        _push(current, m.group(2))
                    else:
                        kind = _classify_section_title(title)
                        if kind in ("s1", "s2", "s3"):
                            current = kind
                    continue
                _push(current, re.sub(r"\s*#{1,6}\s*", " ", piece))
            continue

        if re.match(r"^#{1,6}\s+\S", stripped) or re.match(r"^[一二三四五]、\S", stripped):
            title = re.sub(r"^#{1,6}\s+", "", stripped).strip()
            kind = _classify_section_title(title)
            if kind == "hero" or kind is None:
                continue
            if kind in ("s1", "s2", "s3"):
                current = kind
                continue
            current = "other"
            continue

        # 先清句再判斷，保留「中文＋尾端程式碼殘片」的有用部分
        cleaned = _sanitize_report_sentence(stripped)
        if not cleaned or _is_garbage_content_line(cleaned):
            continue
        letters = re.findall(r"[A-Za-z]", cleaned)
        cjk = re.findall(r"[\u4e00-\u9fff]", cleaned)
        if len(cleaned) >= 24 and len(letters) >= 16 and len(cjk) <= 1:
            continue
        _push(current, cleaned)

    # 未分桶句子依語意再分配
    for ln in buckets["other"]:
        buckets[_sentence_bucket(ln)].append(ln)
    buckets["other"] = []

    # 若幾乎都擠在 s1，再依句子重分
    total = len(buckets["s1"]) + len(buckets["s2"]) + len(buckets["s3"])
    if total and len(buckets["s1"]) >= max(1, total - 1) and not buckets["s2"] and not buckets["s3"]:
        all_sents = []
        for ln in buckets["s1"]:
            all_sents.extend([x.strip() for x in re.split(r"(?<=[。！？])", ln) if x.strip()])
        buckets = {"s1": [], "s2": [], "s3": [], "other": []}
        for s in all_sents:
            buckets[_sentence_bucket(s)].append(s)
        if not buckets["s1"] and all_sents:
            buckets["s1"] = all_sents[:1]
            for s in all_sents[1:]:
                buckets[_sentence_bucket(s)].append(s)

    # 摘要若以「此外／另外」起頭，補一句開場，避免畫面像被截斷
    if buckets["s1"]:
        first = buckets["s1"][0].lstrip("-• ").strip()
        if re.match(r"^(此外|另外|同時|再者|並且)[，,、]?", first):
            buckets["s1"].insert(
                0,
                "依目前監控與控制項狀態，摘要如下。",
            )

    return _build_report_cards(buckets["s1"], buckets["s2"], buckets["s3"])


def _has_report_section_headers(text: str) -> bool:
    """是否含三卡式章節標題（含模型常見變體如「概要」）。"""
    if not text:
        return False
    return bool(
        re.search(
            r"地端\s*LLM\s*智慧合規|"
            r"#{1,6}\s*[一二三123][、.．]\s*(?:事件|不合|合規|風險|修補|具體)|"
            r"^\s*[一二三123][、.．]\s*(?:事件|不合|合規|風險|修補|具體)",
            text,
            re.I | re.M,
        )
    )


def _looks_like_report(text: str) -> bool:
    """判斷是否為（或應整理成）三卡診斷報告。"""
    if not text:
        return False
    if _has_report_section_headers(text):
        return True
    return bool(
        re.search(
            r"##\s*一、\s*事件經過(?:摘要|概要)|"
            r"##\s*二、\s*不合規|"
            r"##\s*三、\s*具體修補建議|"
            r"【報告結束】",
            text,
            re.I,
        )
    )


def _beautify_markdown(text: str, force_report: bool = False) -> str:
    """把鬆散標題整理成 Markdown 層次，利於前端美化。"""
    if not text:
        return text

    # 僅正式三卡報告模式才對齊固定章節標題；聊天模式勿把段落收成三卡
    if force_report:
        replacements = [
            (r"^\s*#{0,4}\s*[*＊]*地端\s*LLM.*?報告[*＊]*\s*[:：]?\s*$", "## 地端 LLM 智慧合規診斷報告"),
            (r"^\s*#{0,4}\s*[*＊]*[一二三123]\s*[\.、．]?\s*事件經過摘要[*＊]*\s*$", "## 一、事件經過摘要"),
            (r"^\s*#{0,4}\s*[*＊]*[一二三123]\s*[\.、．]?\s*小報概述[*＊]*\s*$", "## 一、事件經過摘要"),
            (r"^\s*#{0,4}\s*[*＊]*事變總述[*＊]*\s*$", "## 一、事件經過摘要"),
            (r"^\s*[*＊]*事變總述[*＊]*\s*[:：]?\s*$", "## 一、事件經過摘要"),
            (r"^\s*#{0,4}\s*[*＊]*事件經過摘要[*＊]*\s*$", "## 一、事件經過摘要"),
            (r"^\s*#{0,4}\s*[*＊]*[一二三123]\s*[\.、．]?\s*.*(?:不合規|不符合標準|風險).*[*＊]*\s*$", "## 二、不合規／風險分析"),
            (r"^\s*#{0,4}\s*[*＊]*[一二三123]\s*[\.、．]?\s*.*不合規.*分析[*＊]*\s*$", "## 二、不合規／風險分析"),
            (r"^\s*#{0,4}\s*[*＊]*ISO\s*27001.*(?:分析|解析|評估)[*＊]*\s*$", "## 二、不合規／風險分析"),
            (r"^\s*#{0,4}\s*[*＊]*[一二三123]\s*[\.、．]?\s*.*(?:修補|防護|改善|修正|當事人).*建議[*＊]*\s*$", "## 三、具體修補建議"),
            (r"^\s*#{0,4}\s*[*＊]*具體修補與防護建議[*＊]*\s*$", "## 三、具體修補建議"),
        ]
        for pat, rep in replacements:
            text = re.sub(pat, rep, text, flags=re.I | re.M)

    text = re.sub(r"(可以看到如下[^\n：:]*[:：])\s*", r"\1\n", text)
    # 僅拆「行內」編號（行首 1.／2. 清單保留），且避免把 10.／11. 拆成殘留 0.／1
    text = re.sub(r"(?<=\S)([1-9])[\)）]\s+", r"\n- ", text)
    text = re.sub(
        r"(?<=\S)([1-9])[\.、]\s+(?=[\u4e00-\u9fffA-Za-z])",
        r"\n- ",
        text,
    )
    text = re.sub(r"([^\n])\s*(##\s+)", r"\1\n\n\2", text)
    text = re.sub(r"([。！？；])\s*[-•]\s+", r"\1\n- ", text)
    text = re.sub(r"([^\n])\s+(-\s+[\u4e00-\u9fffA-Za-z])", r"\1\n\2", text)
    text = re.sub(r"(##\s+[^\n]+)\n(?!\n|-)", r"\1\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 只有明確 report 模式才組三卡；自由撰寫時只對齊標題、不重灌模板句
    if force_report and not ENABLE_REPORT_LLM_FREEWRITE:
        return _normalize_report_structure(text)
    return text


def _truncate_output(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """硬性字數上限；優先在句號處截斷。"""
    if not text or len(text) <= limit:
        return text
    # 已是完整三卡 grounded 報告：勿在第二段腰斬（turbo 1050 字太短）
    if (
        re.search(r"##\s*一、", text)
        and re.search(r"##\s*二、", text)
        and re.search(r"##\s*三、", text)
        and ("【報告結束】" in text or "【回答結束】" in text)
        and len(text) <= 2800
    ):
        return text
    cut = text[:limit]
    for sep in ("。", "！", "？", "\n", "；", "."):
        pos = cut.rfind(sep)
        if pos >= int(limit * 0.55):
            cut = cut[: pos + 1]
            break
    cut = cut.rstrip()
    if "【回答結束】" not in cut and "【報告結束】" not in cut:
        marker = "\n【回答結束】"
        if len(cut) + len(marker) > limit:
            cut = cut[: max(0, limit - len(marker))].rstrip()
        cut = cut + marker
    return cut


def _fallback_zh_report(hint: str = "") -> str:
    """淨化後若幾乎無可用繁中，回傳固定可讀模板。"""
    hint = (hint or "").strip()
    hint = re.sub(r"\s+", " ", hint)[:120]
    extra = f"（參考片段）{hint}" if hint and _cjk_count(hint) >= 8 else "請補充日誌後重試。"
    return _normalize_report_structure(
        "## 地端 LLM 智慧合規診斷報告\n"
        "## 一、事件經過摘要\n"
        f"模型先前輸出格式異常，已自動重整。{extra}\n"
        "## 二、不合規／風險分析\n"
        "- 目前無法完成可靠合規判定。\n"
        "## 三、具體修補建議\n"
        "- 請改以單一控制項與精簡日誌重新分析。\n"
        "【報告結束】"
    )


def _demote_report_to_chat(text: str) -> str:
    """聊天模式若誤出三卡，改成一般段落，不渲染報告卡。"""
    if not text:
        return text
    t = text
    t = re.sub(r"^\s*#{0,6}\s*.*智慧合規.*報告.*$", "", t, flags=re.M)
    t = re.sub(r"^\s*#{0,6}\s*[一二三][、.．][^\n]*$", "", t, flags=re.M)
    t = re.sub(
        r"^\s*#{0,6}\s*(?:事件經過(?:摘要|概要)|不合規|合規項目|具體修補)[^\n]*$",
        "",
        t,
        flags=re.I | re.M,
    )
    t = re.sub(
        r"^\s*[一二三123][、.．]\s*(?:事件經過(?:摘要|概要)|不合規.*|合規.*|具體修補建議)\s*$",
        "",
        t,
        flags=re.M,
    )
    t = re.sub(r"【(?:報告|回答)結束】", "", t)
    # 去掉三卡預設填空句，避免閒聊被拼成假報告
    t = re.sub(
        r"^\s*(?:[-*•]\s*)?(?:依目前監控資料，尚未觀察到[^\n]*|"
        r"目前無明確高風險不合規證據[^\n]*|"
        r"維持現有監控與定期複核[^\n]*|"
        r"請交叉比對控制項量化指標[^\n]*)\s*$",
        "",
        t,
        flags=re.M,
    )
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def normalize_llm_output(text: str, mode: str = "report") -> str:
    """統一後處理：去日文/亂碼 → 繁中 → 排版 → 字數上限（圖表不計入）。

    mode:
      - report: 診斷三卡（監控分析／明確要求報告時）
      - chat: 一般對話，不強制、也不渲染報告模板
    """
    if not text:
        return text
    mode = (mode or "report").lower()
    force_report = mode == "report" and ENABLE_THREE_CARD_REPORT
    out_limit = MAX_OUTPUT_CHARS_REPORT if force_report else MAX_OUTPUT_CHARS

    visual_blocks = re.findall(r"```(?:chart|mermaid)[\s\S]*?```", text, flags=re.I)
    body = re.sub(r"```(?:chart|mermaid)[\s\S]*?```", "", text, flags=re.I)

    body = _collapse_repetitions(body)
    body = _strip_code_fences(body)  # 先拿掉 code fence，避免拆成一堆卡
    body = _strip_context_leak(body)  # 先去掉照抄的 RAG／Log
    body = _strip_garbage(body)
    body = _to_traditional(body)
    body = _drop_japanese_lines(body)  # 轉繁後再清一次
    body = _strip_context_leak(body)  # 再清一次殘留前綴
    if force_report:
        body = _beautify_markdown(body, force_report=True)
    else:
        # 聊天：即使模型亂出報告標題，也不強制三卡
        if _has_report_section_headers(body) or _looks_like_report(body):
            body = _demote_report_to_chat(body)
        body = _beautify_markdown(body, force_report=False)
        if _has_report_section_headers(body) or _looks_like_report(body):
            body = _demote_report_to_chat(body)
    body = _collapse_repetitions(body)
    body = _sanitize_domain_text(body)
    # 完整三卡（含 grounded 現況）勿被 turbo 字數砍掉第三段
    if (
        force_report
        and _looks_like_report(body)
        and "【報告結束】" in body
        and "## 三、" in body
    ):
        out_limit = max(out_limit, 2200)
    body = _truncate_output(body, out_limit)
    body = (body or "").replace("\ufffd", "")
    # 字數截斷後若停在半句，收成完整句
    if body and _looks_truncated(body):
        body = _finish_incomplete_sentence(body)

    # 仍含日文或中文過少：盡量保留可用繁中，避免丟出「輸出異常」空話
    if _needs_zh_retry(body):
        zh_lines = [
            ln.strip() for ln in body.splitlines()
            if ln.strip()
            and not _has_japanese(ln)
            and _cjk_count(ln) >= 4
            and not re.match(r"^#{1,6}\s*$", ln.strip())
        ]
        if force_report:
            body = _fallback_zh_report(" ".join(zh_lines[:3]))
        elif zh_lines:
            body = "\n".join(zh_lines[:10])
        elif _cjk_count(body) >= 12:
            # 仍有中文但被拆行條件濾掉：保留去日文後的正文
            body = _drop_japanese_lines(body)
            body = re.sub(r"\n{3,}", "\n\n", body).strip()
        elif ENABLE_UNSTABLE_FORMAT_FALLBACK:
            body = (
                "目前這則回答格式不穩定，已略過無效內容。\n\n"
                "你可以改問：\n"
                "- 目前合規現況\n"
                "- 幫我說明修補步驟\n"
                "- 貼上 Cisco syslog 請我分析"
            )
        else:
            # 關閉罐頭提示：盡量保留模型原文
            body = _drop_japanese_lines(body)
            body = re.sub(r"\n{3,}", "\n\n", body).strip() or body.strip()

    if visual_blocks:
        seen = set()
        uniq = []
        for b in visual_blocks:
            key = b.strip()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(key)
        body = body.rstrip() + "\n\n" + "\n\n".join(uniq)
    if mode == "chat":
        body = _polish_chat_output(body)
    return body.strip()


def _finish_incomplete_sentence(text: str) -> str:
    """若結尾明顯半截，收束到完整句，避免畫面停在半字／�。"""
    if not text:
        return text
    t = text.replace("\ufffd", "").rstrip()
    # 去掉模型角色標籤殘留
    t = re.sub(
        r"^(?:system|assistant|user|System|Assistant|User)\s*[:：]?\s*\n+",
        "",
        t,
    )
    t = re.sub(r"<\|(?:system|assistant|user|end|im_start|im_end)\|>\s*", "", t)
    t = t.strip()
    if not t:
        return text.strip()
    if t.endswith(("。", "！", "？", "】", "`")):
        return t
    # 去掉尾端半截英數／殘破標點
    t = re.sub(r"[\sA-Za-z0-9_\-\'\"\.]+$", "", t).rstrip("，,、:：；;…-—")
    # 仍半截：退回上一句完整句號，避免「……回�」這種斷尾
    if t and t[-1] not in "。！？.!?」』）)】" and _looks_truncated(t):
        m = re.search(r"^(.*[。！？])", t, flags=re.S)
        if m and _cjk_count(m.group(1)) >= 20:
            t = m.group(1).rstrip()
        elif not t.endswith(("。", "！", "？")):
            t += "。"
    elif t and not t.endswith(("。", "！", "？")):
        t += "。"
    return t


class _StopOnStrings(StoppingCriteria):
    """生成出現結束標記時立刻停止，避免空轉到 max_new_tokens。"""

    def __init__(self, stop_ids_list, start_len):
        self.stop_ids_list = stop_ids_list or []
        self.start_len = start_len

    def __call__(self, input_ids, scores, **kwargs):
        if not self.stop_ids_list:
            return False
        seq = input_ids[0].tolist()
        gen = seq[self.start_len:]
        for stop_ids in self.stop_ids_list:
            n = len(stop_ids)
            if n and len(gen) >= n and gen[-n:] == stop_ids:
                return True
        return False


def _reset_chat_template_probe() -> None:
    """切換模型後重探 chat template（Gemma 不支援 system role）。"""
    global _llm_supports_system_role
    _llm_supports_system_role = None


def _chat_template_supports_system() -> bool:
    """探測目前 tokenizer 是否可在 apply_chat_template 使用 system role。"""
    global _llm_supports_system_role
    if _llm_supports_system_role is not None:
        return _llm_supports_system_role
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        _llm_supports_system_role = False
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
        _llm_supports_system_role = True
    except Exception as e:
        msg = str(e).lower()
        if "system role" in msg or "system" in msg:
            print("ℹ️ 此模型 chat template 不支援 system role → 推論時併入 user")
        _llm_supports_system_role = False
    return _llm_supports_system_role


def _coalesce_chat_turns(messages: list[dict]) -> list[dict]:
    """
    合併連續相同 role，滿足 Gemma 等模板 user/assistant 嚴格交替。
    """
    if not messages:
        return messages
    cleaned: list[dict] = []
    for m in messages:
        role = (m.get("role") or "user").lower()
        if role not in ("user", "assistant"):
            role = "user"
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if cleaned and cleaned[-1]["role"] == role:
            cleaned[-1]["content"] = (
                f"{cleaned[-1]['content']}\n\n{content}".strip()
            )
        else:
            cleaned.append({"role": role, "content": content})
    if cleaned and cleaned[0]["role"] == "assistant":
        cleaned.insert(0, {"role": "user", "content": "（續前對話）"})
    if cleaned and cleaned[-1]["role"] == "assistant":
        cleaned.append({"role": "user", "content": "請依上文繼續回答。"})
    return cleaned


def _normalize_chat_messages(messages: list[dict]) -> list[dict]:
    """
    Gemma 等模型：將 system 提示併入 user，並確保 user/assistant 交替。
    """
    if not messages:
        return messages
    if _chat_template_supports_system():
        return _coalesce_chat_turns([
            {"role": (m.get("role") or "user"), "content": m.get("content") or ""}
            for m in messages
        ])

    system_parts: list[str] = []
    out: list[dict] = []
    for m in messages:
        role = (m.get("role") or "user").lower()
        content = (m.get("content") or "").strip()
        if role == "system":
            if content:
                system_parts.append(content)
            continue
        if role not in ("user", "assistant"):
            role = "user"
        out.append({"role": role, "content": content})

    if system_parts:
        system_blob = "\n\n".join(system_parts).strip()
        merged = False
        for i, m in enumerate(out):
            if m["role"] == "user":
                body = m.get("content") or ""
                out[i] = {
                    "role": "user",
                    "content": f"{system_blob}\n\n{body}".strip() if body else system_blob,
                }
                merged = True
                break
        if not merged:
            out.insert(0, {"role": "user", "content": system_blob})

    if not out:
        return [{"role": "user", "content": "\n\n".join(system_parts).strip() or "你好"}]

    return _coalesce_chat_turns(out)


def _llm_is_ready() -> bool:
    if USE_OLLAMA:
        return ollama_service.is_ready()
    return model is not None and tokenizer is not None


def _looks_like_ollama_meta_leak(text: str) -> bool:
    """Ollama 小模型把 system／續寫指示當成回答。"""
    t = (text or "").strip()
    if not t:
        return False
    return bool(
        re.search(
            r"分析使用者|使用者本則問題|從斷點|若有對話紀錄|"
            r"作為\s*AI|上一則無效|禁止複述系統|首句寒暄|本則意圖|"
            r"【多輪對話】|請從斷點|剛才回答未完成|"
            r"^首先[，,]?\s*用[户戶]|^首先[，,]?\s*用户|"
            r"這是一個友好的打招呼|必須依上文理解|"
            r"但指示說|指示說|可能的回應|所以我不需要假設",
            t,
            re.I | re.M,
        )
    )


def _sanitize_ollama_chat_reply(text: str) -> str:
    """去掉 Ollama 聊天中的 meta／推理／prompt 殘留。"""
    if not text:
        return ""
    t = (text or "").replace("\ufffd", "").strip()
    if _looks_like_ollama_meta_leak(t):
        kept = []
        for ln in t.splitlines():
            s = ln.strip()
            if not s or _looks_like_ollama_meta_leak(s):
                continue
            if _is_prompt_instruction_line(s):
                continue
            kept.append(s)
        t = "\n".join(kept).strip()
    else:
        kept = []
        for ln in t.splitlines():
            s = ln.strip()
            if not s or _is_prompt_instruction_line(s):
                continue
            kept.append(s)
        t = "\n".join(kept).strip() if kept else t
    t = re.sub(r"^(?:system|assistant|user)\s*[:：]?\s*", "", t, flags=re.I)
    if _looks_like_ollama_meta_leak(t):
        return ""
    return t.strip()


def _run_ollama_llm(messages, max_new_tokens, allow_continue, output_mode):
    """Ollama HTTP 推論（聊天走單次呼叫，避免續寫污染）。"""
    if not ollama_service.is_ready():
        err = ollama_service.current_info().get("ollama_error") or "Ollama 未連線"
        return f"⚠️ {err}。請先執行 ollama serve。"

    msgs = _normalize_chat_messages(messages)
    token_budget = int(max_new_tokens or MAX_NEW_TOKENS)
    if output_mode == "chat":
        token_budget = min(token_budget, MAX_NEW_TOKENS)
    else:
        token_budget = min(token_budget, MAX_NEW_TOKENS_AUDIT)

    with _llm_lock:
        print(
            f"🦙 Ollama chat | model={ollama_service.current_model_name()} "
            f"max_tokens={token_budget} mode={output_mode}"
        )
        reply = ollama_service.chat(msgs, max_new_tokens=token_budget)
        reply = _sanitize_ollama_chat_reply(_collapse_repetitions(reply or ""))

        if output_mode == "chat":
            if not reply or _looks_like_ollama_meta_leak(reply):
                print("⚠️ Ollama 聊天輸出異常，重答一次（簡短）…")
                retry = ollama_service.chat(
                    msgs,
                    max_new_tokens=min(token_budget, 180),
                )
                reply = _sanitize_ollama_chat_reply(_collapse_repetitions(retry or ""))
            if not reply:
                return "您好！我是 Semi-Shield Cyber Agent，請問有什麼可以協助您？"
            return _polish_chat_output(normalize_llm_output(reply, mode="chat"))

        if not reply:
            return "模型未回傳有效內容，請稍後再試或改選 qwen2.5:3b。"
        cleaned = normalize_llm_output(reply, mode=output_mode)
        return (cleaned or "").replace("\ufffd", "")


def run_llm(messages, max_new_tokens=MAX_NEW_TOKENS, allow_continue=False, output_mode="report"):
    """通用 LLM 推論：接收 chat messages，回傳模型文字回應。"""
    output_mode = (output_mode or "report").lower()

    if USE_OLLAMA:
        return _run_ollama_llm(messages, max_new_tokens, allow_continue, output_mode)

    if model is None or tokenizer is None:
        return "模型未成功載入，無法提供 AI 診斷。"

    # CPU：縮短生成、關閉續寫，體感會快很多
    if not _use_cuda_for_llm():
        max_new_tokens = min(int(max_new_tokens or MAX_NEW_TOKENS), int(MAX_NEW_TOKENS))
        if output_mode == "chat":
            max_new_tokens = min(max_new_tokens, 96)
            allow_continue = False
        else:
            max_new_tokens = min(max_new_tokens, int(MAX_NEW_TOKENS_AUDIT))
            allow_continue = False

    # 序列化 GPU 推論，避免 Flask 多執行緒並發 generate 造成 CUDA illegal access
    with _llm_lock:
        return _run_llm_locked(
            messages, max_new_tokens, allow_continue, output_mode, False
        )


def _run_llm_locked(
    messages, max_new_tokens, allow_continue, output_mode, _cpu_failover_done=False
):
    if model is None or tokenizer is None:
        return "模型未成功載入，無法提供 AI 診斷。"

    try:
        def _generate_once(msgs, token_budget, *, safe_mode=False):
            msgs = _normalize_chat_messages(msgs)
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    prompt = tokenizer.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception as tpl_err:
                    err_msg = str(tpl_err).lower()
                    if "system" in err_msg or "alternate" in err_msg or "roles must" in err_msg:
                        global _llm_supports_system_role
                        _llm_supports_system_role = False
                        msgs = _normalize_chat_messages(msgs)
                        prompt = tokenizer.apply_chat_template(
                            msgs,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    else:
                        raise
            else:
                parts = []
                for msg in msgs:
                    parts.append(f"<|{msg['role']}|>\n{msg['content']}<|end|>\n")
                parts.append("<|assistant|>\n")
                prompt = "".join(parts)

            # 保留尾端（使用者問題 + generation prompt），避免長 system 把問題截掉
            prev_side = getattr(tokenizer, "truncation_side", "right")
            prompt_cap = MAX_PROMPT_TOKENS
            if safe_mode:
                prompt_cap = min(prompt_cap, 640)
                token_budget = min(int(token_budget), 160)
            try:
                tokenizer.truncation_side = "left"
                local_inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=prompt_cap,
                )
            finally:
                tokenizer.truncation_side = prev_side
            device = next(model.parameters()).device
            local_inputs = {
                k: v.contiguous().to(device, non_blocking=False)
                for k, v in local_inputs.items()
            }

            input_token_len = local_inputs["input_ids"].shape[1]
            print(
                f"🧮 prompt_tokens={input_token_len}, max_new_tokens={token_budget}"
                + (" [safe]" if safe_mode else "")
            )

            stopping = StoppingCriteriaList([
                _StopOnStrings(_stop_ids_list, input_token_len)
            ])

            gen_kwargs = dict(
                max_new_tokens=token_budget,
                do_sample=False,
                use_cache=not safe_mode,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=stopping,
            )
            # Phi 加強防重複；Qwen 聊天也開輕量 penalty（turbo 不再完全關掉）
            phi = _is_phi_model_active()
            penalty = 1.18 if phi else max(float(REPETITION_PENALTY or 1.0), 1.08)
            ngram = 4 if phi else (
                NO_REPEAT_NGRAM or (3 if output_mode == "chat" else 0)
            )
            if penalty and penalty != 1.0:
                gen_kwargs["repetition_penalty"] = penalty
            if ngram and ngram > 0 and not safe_mode:
                gen_kwargs["no_repeat_ngram_size"] = ngram

            with torch.inference_mode():
                local_outputs = model.generate(**local_inputs, **gen_kwargs)

            new_tokens = local_outputs[0][input_token_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            del local_outputs, local_inputs
            return _collapse_repetitions(text)

        def _generate_with_cuda_retry(msgs, token_budget):
            try:
                return _generate_once(msgs, token_budget, safe_mode=False)
            except Exception as gen_err:
                if not _is_cuda_fault(gen_err):
                    raise
                # illegal access 後 GPU 已不可用：勿再 GPU 重試，直接切 CPU
                print("⚠️ CUDA 異常，停止 GPU 重試，改切 CPU 後備…")
                if not _failover_llm_to_cpu(str(gen_err)):
                    raise
                return _generate_once(
                    msgs,
                    min(int(token_budget), 160),
                    safe_mode=True,
                )

        reply = _generate_with_cuda_retry(messages, max_new_tokens)
        reply = _collapse_repetitions(reply)
        reply = reply.replace("\ufffd", "")
        reply = re.sub(
            r"^(?:system|assistant|user)\s*[:：]?\s*\n+",
            "",
            reply,
            flags=re.I,
        )

        # 條列循環／訓練殘留：強制短重寫一次
        if output_mode == "chat" and (
            _looks_like_list_loop(reply)
            or _looks_like_train_leak(reply)
            or (_is_phi_model_active() and reply.count("具體") >= 8)
            or _cjk_count(reply) < 12
        ):
            print("⚠️ 偵測到無效輸出／訓練殘留，強制重答一次…")
            retry_messages = list(messages) + [
                {
                    "role": "user",
                    "content": (
                        "上一則無效（亂碼、訓練殘留或條列循環）。"
                        "請用 4-8 句繁體中文直接回答使用者問題，"
                        "禁止輸出「使用者：…」、禁止複述系統指示；最後【回答結束】。"
                    ),
                }
            ]
            retry = _generate_with_cuda_retry(
                retry_messages,
                min(max_new_tokens, 240 if SPEED_MODE == "turbo" else 320),
            )
            retry = _collapse_repetitions(retry or "")
            if (
                retry
                and not _looks_like_list_loop(retry)
                and not _looks_like_train_leak(retry)
                and _cjk_count(retry) >= 24
            ):
                reply = retry
            elif retry and _cjk_count(retry) > _cjk_count(reply):
                reply = retry

        # 續寫最多 1 次；聊天在 turbo 也允許短續寫，避免半句／�
        do_continue = allow_continue or output_mode == "chat"
        if do_continue and _looks_truncated(reply) and not _looks_like_train_leak(reply):
            print("⚠️ 偵測到回答可能被截斷，自動續寫補完（1 次）...")
            clipped = _collapse_repetitions(reply)
            cont_budget = (
                min(180, max_new_tokens // 2 + 64)
                if SPEED_MODE == "turbo"
                else min(220, max_new_tokens // 2 + 64)
            )
            continue_messages = list(messages) + [
                {"role": "assistant", "content": clipped[-800:]},
                {
                    "role": "user",
                    "content": (
                        "剛才的回答未完成。請不要重頭開始，也不要重複同一句。"
                        "用繁體中文從斷點簡潔補完（最多 5 句），最後一行寫【回答結束】。"
                    ),
                },
            ]
            continuation = _generate_with_cuda_retry(continue_messages, cont_budget)
            continuation = _collapse_repetitions(continuation or "").replace("\ufffd", "")
            # 續寫若變成亂碼／訓練殘留則捨棄
            if continuation and not _looks_like_train_leak(continuation) and _cjk_count(continuation) >= 8:
                if continuation.startswith(clipped[:24]):
                    reply = continuation
                else:
                    reply = (clipped.rstrip() + "\n" + continuation.lstrip()).strip()
                reply = _collapse_repetitions(reply)

        if _looks_truncated(reply):
            reply = _finish_incomplete_sentence(reply)

        cleaned = normalize_llm_output(reply, mode=output_mode)
        cleaned = (cleaned or "").replace("\ufffd", "")
        cleaned = re.sub(
            r"^(?:system|assistant|user)\s*[:：]?\s*\n+",
            "",
            cleaned,
            flags=re.I,
        )

        # 不合格才重寫（含 turbo 聊天：訓練殘留／幾乎無中文）
        need_retry = (
            _has_japanese(cleaned)
            or "已自動攔截" in cleaned
            or _needs_zh_retry(cleaned)
        )
        if need_retry:
            print("⚠️ 偵測到非繁中／日文混雜，強制以繁體中文重寫一次...")
            if output_mode == "chat":
                rewrite_hint = (
                    "【強制重寫】上一則無效。請只用繁體中文、像一般對話簡短重答，"
                    "禁止日文，禁止輸出「智慧合規診斷報告」三卡格式；≤400字；最後【回答結束】。"
                )
            else:
                if ENABLE_REPORT_LLM_FREEWRITE:
                    rewrite_hint = (
                        "【強制重寫】上一則無效。請整份只用繁體中文重寫診斷報告，"
                        "禁止日文；依本次日誌／計數自由論述摘要、風險與建議；"
                        "勿套罐頭句；≤1260字；最後一行【報告結束】。"
                    )
                else:
                    rewrite_hint = (
                        "【強制重寫】上一則無效。請整份只用繁體中文重寫，"
                        "禁止日文／日本語／ひらがな／カタカナ，禁止兩種語言各寫一遍。"
                        "嚴格使用：## 地端 LLM 智慧合規診斷報告、## 一、事件經過摘要、"
                        "## 二、不合規／風險分析、## 三、具體修補建議；條列精簡；≤1260字；"
                        "最後一行【報告結束】。"
                    )
            retry_messages = list(messages) + [
                {"role": "user", "content": rewrite_hint}
            ]
            retry_raw = _generate_with_cuda_retry(
                retry_messages,
                min(max_new_tokens, 280 if SPEED_MODE == "turbo" else 480),
            )
            retry_clean = normalize_llm_output(retry_raw, mode=output_mode)
            if "已自動攔截" not in retry_clean and not _has_japanese(retry_clean):
                if _cjk_count(retry_clean) >= 40:
                    cleaned = retry_clean
            elif (
                "已自動攔截" in cleaned
                and "已自動攔截" not in retry_clean
                and _cjk_count(retry_clean) > 40
            ):
                cleaned = retry_clean

        return cleaned

    except torch.cuda.OutOfMemoryError:
        print("⚠️ 發生 CUDA Out of Memory！改切 CPU 後備…")
        if (not _cpu_failover_done) and _failover_llm_to_cpu("OOM"):
            return _run_llm_locked(
                messages,
                min(int(max_new_tokens), 160),
                False,
                output_mode,
                True,
            )
        return "分析失敗：GPU 顯存不足，且無法載入 CPU 後備模型。請設 FORCE_CPU=1 後重啟。"

    except Exception as e:
        print(f"LLM 推理發生錯誤: {str(e)}")
        if _is_cuda_fault(e) and (not _cpu_failover_done):
            if _failover_llm_to_cpu(str(e)):
                return _run_llm_locked(
                    messages,
                    min(int(max_new_tokens), 160),
                    False,
                    output_mode,
                    True,
                )
            return (
                "分析過程發生 GPU 異常（CUDA）。"
                "請關閉後端，用 run_edge.bat 或設 FORCE_CPU=1 後重啟。"
            )
        if _is_cuda_fault(e):
            return (
                "GPU 已損壞且 CPU 重試失敗。"
                "請關閉後端後用 FORCE_CPU=1 或 run_edge.bat 重啟。"
            )
        return f"分析過程發生異常：{str(e)}"


def is_empty_log(log_text):
    """判斷是否為無可用事件資料。"""
    if not log_text:
        return True
    text = str(log_text).strip()
    empty_markers = ("無資料", "沒有相關事件", "尚無可用", "暫無")
    return any(m in text for m in empty_markers)


def _has_real_log_signal(log_text: str) -> bool:
    """日誌是否含可用的設備事件（Cisco / 合規相關）。"""
    t = log_text or ""
    if is_empty_log(t):
        return False
    return bool(
        re.search(
            r"%[A-Z0-9_-]+-\d+-[A-Z0-9_]+|SEC_LOGIN|CONFIG_I|LOGIN_FAILED|"
            r"RADIUS|TACACS|SNMP-|CRYPTO|ILPOWER|UPDOWN|PORT_SECURITY",
            t,
            re.I,
        )
    )


def _report_lacks_domain_content(text: str) -> bool:
    """報告是否缺少實際稽核語意（只剩標題／預設句／提示詞）。"""
    if not text:
        return True
    body = re.sub(r"(?m)^#{1,6}\s+.*$", "", text)
    body = re.sub(r"【(?:回答|報告)結束】", "", body)
    body = re.sub(
        r"依目前監控資料，尚未觀察到|目前無明確高風險|維持現有監控與定期複核|"
        r"交叉比對控制項量化指標|本段暫無足夠資料",
        "",
        body,
    )
    kept = []
    for ln in body.splitlines():
        ln = ln.strip()
        if not ln or _is_prompt_instruction_line(ln) or _is_garbage_content_line(ln):
            continue
        kept.append(ln)
    joined = "\n".join(kept)
    if _cjk_count(joined) < 24:
        return True
    return not re.search(
        r"登入|組態|介面|驗證|RADIUS|SNMP|弱點|風險|不合規|CAPA|PoE|"
        r"供電|日誌|監控|修補|帳號|來源\s*IP|控制項",
        joined,
        re.I,
    )


def _audit_report_has_defects(text: str, log_text: str = "", control_key: str | None = None) -> bool:
    """稽核報告常見小模型缺陷：空白占位、省略語、登入語意矛盾、空泛模板。"""
    t = text or ""
    if not t.strip():
        return True
    if re.search(
        r"此處省略|省略了詳細|且出現了\s*的訊息|SSH\s*埠\s*\(\s*\)|"
        r"依目前監控數據，未觀察到需立即升級|尚未觀察到需立即升級|"
        r"目前無明確高風險不合規證據.*交叉比對|維持現有監控與定期複核.*CAPA",
        t,
        re.I | re.S,
    ):
        return True
    # 摘要寫成功登入，風險段卻寫暴力破解／大量失敗
    if re.search(r"LOGIN_SUCCESS|登入成功|成功登入", t, re.I) and re.search(
        r"登入失敗|暴力破解|credential\s*stuff|失敗次數|brute", t, re.I
    ):
        if not re.search(r"LOGIN_FAILED|AUTHFAIL|denied|拒絕", log_text or "", re.I):
            return True
    # 有 fail/review 指標卻全段寫「無風險／無重大事件」
    if re.search(r"status\s*=\s*fail|status=fail", log_text or "", re.I):
        if re.search(r"未觀察到.*重大|相對良好|無明確高風險|維持現狀即可", t, re.I):
            if not re.search(r"fail|偏高|需稽核|優先", t, re.I):
                return True
    return False


def _report_is_weak(text: str) -> bool:
    """判斷三卡報告是否幾乎沒有實質內容或遭程式碼／提示詞污染。"""
    if not text:
        return True
    # 明顯洩漏／離題／提示詞回聲
    if re.search(
        r"```|control_id\s*:|respond_to\?|is_string|MyClass|我想知道如何使用|"
        r"please let me know|further assistance|\[REPORT\s*END\]|"
        r"格式撰寫|撰寫診斷|必要的資訊和建議|合作規診斷|"
        r"開始輸入答案|開始輸出答案|開始作答|輸入答案|"
        r"智慧合規診斷報告.{0,12}開始|此處省略|且出現了\s*的訊息",
        text,
        re.I,
    ):
        return True
    body = re.sub(r"(?m)^#{1,6}\s+.*$", "", text)
    body = re.sub(r"【(?:回答|報告)結束】", "", body)
    body = re.sub(
        r"依目前監控資料，尚未觀察到|目前無明確高風險|維持現有監控與定期複核|交叉比對控制項量化指標",
        "",
        body,
    )
    useful = []
    for ln in body.splitlines():
        ln = ln.strip()
        if not ln or _is_garbage_content_line(ln) or _is_prompt_instruction_line(ln):
            continue
        if _cjk_count(ln) >= 8 and "本段暫無" not in ln and "請補充日誌" not in ln:
            useful.append(ln)
    # 三段若高度重複同一句，也視為弱輸出
    uniq = {re.sub(r"\s+", "", u) for u in useful}
    if len(useful) >= 2 and len(uniq) <= 1:
        return True
    if _report_lacks_domain_content(text):
        return True
    return len(useful) < 2 or _cjk_count("".join(useful)) < 36


def _extract_cisco_events(log_text: str) -> list[dict]:
    """從文字抽出 Cisco %FACILITY-SEV-MNEMONIC 事件。"""
    events = []
    for line in (log_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.search(
            r"%([A-Z0-9_-]+)-(\d+)-([A-Z0-9_]+):\s*(.*)$",
            line,
            re.I,
        )
        if not m:
            continue
        fac, sev, mne, body = (
            m.group(1).upper(),
            m.group(2),
            m.group(3).upper(),
            m.group(4).strip(),
        )
        events.append({
            "facility": fac,
            "severity": sev,
            "mnemonic": mne,
            "body": body,
            "code": f"%{fac}-{sev}-{mne}",
            "raw": line,
        })
    return events


def _devices_from_events(events: list[dict]) -> str:
    names = []
    for e in events:
        m = re.search(r"\[([^\]]+)\]", e.get("raw") or "")
        if m:
            names.append(m.group(1).strip())
    names = sorted(set(names))
    return "、".join(names) if names else "Cisco 交換器"


def _sample_raw_lines(events: list[dict], n: int = 3) -> list[str]:
    out = []
    for e in events[:n]:
        raw = (e.get("raw") or e.get("body") or "")[:160]
        if raw:
            out.append(raw)
    return out


def build_cisco_log_grounded_report(
    log_text,
    control_title=None,
    metric_summary=None,
) -> str | None:
    """
    依日誌原文產生較完整三卡診斷（不靠 LLM）。
    用於壓制微調模型常見幻覺，並提供可執行的排查步驟。
    """
    events = _extract_cisco_events(log_text)
    if not events:
        return None

    title = control_title or "ISO 27001 控制項"
    metric = (metric_summary or "").strip()
    codes = sorted({e["code"] for e in events})
    updown = [
        e for e in events
        if e["mnemonic"] == "UPDOWN" or "UPDOWN" in e["code"]
    ]
    login_ok = [e for e in events if "LOGIN_SUCCESS" in e["mnemonic"]]
    login_fail = [e for e in events if "LOGIN_FAILED" in e["mnemonic"]]
    logout = [
        e for e in events
        if e["mnemonic"] in ("LOGOUT", "TTY_EXPIRE_TIMER") or "LOGOUT" in e["mnemonic"]
    ]
    snmp = [
        e for e in events
        if e["facility"] == "SNMP"
        or re.search(r"SNMP|CRYPTO|PKI|IKE|CERTIFICATE", e["code"], re.I)
    ]
    malware = [
        e for e in events
        if re.search(
            r"MALWARE|IPS|PSECURE|PORTSCAN|HOST_ATTACK|AV_ALERT|USB_DEVICE",
            e["code"] + e["body"],
            re.I,
        )
    ]
    config = [
        e for e in events
        if re.search(r"CONFIG_I|PARSER|CMD_DENIED", e["code"], re.I)
    ]
    patch = [
        e for e in events
        if re.search(r"BOOT|INSTALL|IOSXE|PLATFORM|IMAGE|UPGRADE|SMU", e["code"], re.I)
    ]
    supplier = [
        e for e in events
        if re.search(r"CDP|LLDP|PNP_|LOGGINGHOST|NEIGHBOR|BGP", e["code"], re.I)
    ]
    radius = [
        e for e in events
        if re.search(r"RADIUS|TACACS|AAA|DOT1X|MAB_", e["code"], re.I)
    ]
    power = [
        e for e in events
        if re.search(r"ILPOWER|ERR_DISABLE|SPANTREE", e["code"], re.I)
    ]

    if updown:
        ifaces = []
        for e in updown:
            im = re.search(r"Interface\s+([A-Za-z0-9/.-]+)", e["body"], re.I)
            if im:
                ifaces.append(im.group(1))
        iface_set = sorted(set(ifaces))
        iface_txt = "、".join(iface_set) or "相關埠位"
        down_n = sum(
            1 for e in updown if re.search(r"changed state to down", e["body"], re.I)
        )
        up_n = sum(
            1 for e in updown if re.search(r"changed state to up", e["body"], re.I)
        )
        link_n = sum(1 for e in updown if e["facility"] == "LINK")
        proto_n = sum(1 for e in updown if "LINEPROTO" in e["facility"])
        dev_txt = _devices_from_events(updown)
        samples = _sample_raw_lines(updown, 3)
        severity = (
            "高" if len(updown) >= 6 or len(iface_set) >= 2
            else ("中" if len(updown) >= 3 else "低～中")
        )
        s1 = [
            f"控制項「{title}」：於 {dev_txt} 觀測到 {len(updown)} 筆介面狀態變更"
            f"（LINK {link_n}／LINEPROTO {proto_n}；訊息碼 {', '.join(codes[:4])}）。",
            f"涉及埠位：{iface_txt}。精確計數約 down×{down_n}、up×{up_n}，"
            f"判定為 **link flap（介面抖動）**，嚴重度評估：{severity}。",
            "事件屬實體層／二層連線可用性問題；不是組態檔增刪，不是 MFG 配方變更，也不是重放攻擊。",
        ]
        if samples:
            s1.append("代表性原文：" + " ｜ ".join(samples[:2]))
        if metric:
            s1.append(f"控制項量化：{metric[:200]}")
        s2 = [
            "合規對應：可掛 A.8.19 做營運／變更事件稽核；本質風險是製程網可用性與通訊完整性。",
            "對半導體廠影響：機台／AMHS／廠務設備乙太網中斷可能造成 recipe 傳輸失敗、警報漏報或生產暫停。",
            "常見根因：線材鬆脫／氧化、SFP／光模組故障、對端設備重啟、PoE 電力不足、"
            "speed/duplex 協商失敗、UDLD 單向連線、生成樹／errdisable 後恢復。",
            "若同一埠位短時間反覆 up/down，應優先當硬體／實體問題排查，而非資安攻擊劇本。",
            "需排除維護窗口內的合法拔線；無變更單的反覆抖動應升級為事件。",
        ]
        s3 = [
            f"現場：檢查 {iface_txt} 接頭、跳線、模組與對端設備狀態指示燈。",
            "交換器指令建議：show interface <埠>、show errdisable recovery、"
            "show power inline（若 PoE）、確認 speed/duplex／udld。",
            "監控：對同一 Interface 設定「N 分鐘內 UPDOWN ≥ 閾值」告警，並保留原始 syslog。",
            "流程：維護拔線須填變更窗口；復原後做 15～30 分鐘觀察確認不再抖動。",
            "若抖動伴隨 errdisable／CRC 上升：更換線材或模組，必要時先隔離測試埠。",
        ]
        return _build_report_cards(s1, s2, s3)

    if login_ok or login_fail or logout or radius:
        ports, users, sources = set(), set(), set()
        for e in login_ok + login_fail:
            pm = re.search(r"localport:\s*(\d+)", e["body"], re.I)
            um = re.search(r"user:\s*([^\]]+)", e["body"], re.I)
            sm = re.search(r"Source:\s*([\d.]+)", e["body"], re.I)
            if pm:
                ports.add(pm.group(1))
            if um:
                users.add(um.group(1).strip())
            if sm:
                sources.add(sm.group(1))
        port_txt = "、".join(sorted(ports)) or "未知"
        user_txt = "、".join(sorted(users)) or "未知"
        src_txt = "、".join(sorted(sources)) or "未知"
        telnet = "23" in ports
        ssh = "22" in ports
        shared = bool(re.search(r"\bcisco\b|admin|shared", user_txt, re.I))
        samples = _sample_raw_lines(login_ok + login_fail + logout + radius, 3)
        s1 = [
            f"控制項「{title}」：管理面存取生命週期——"
            f"成功 {len(login_ok)}、失敗 {len(login_fail)}、結束／逾時 {len(logout)}、"
            f"AAA／RADIUS 相關 {len(radius)}。",
            f"觀測帳號：{user_txt}；來源 IP：{src_txt}；localport：{port_txt}。",
            "此為交換器／網路設備管理登入事件，不是現場 PLC 控制指令，也不是重放攻擊。",
        ]
        if samples:
            s1.append("代表性原文：" + " ｜ ".join(samples[:2]))
        if metric:
            s1.append(f"控制項量化：{metric[:200]}")
        s2 = [
            "合規對應：ISO 27001 A.7.4／A.11.2（存取控制與身分驗證）。",
            "究責風險："
            + (
                "使用共用／預設風格帳號（如 cisco）時，無法對到個人，稽核常列 Minor／Major NC。"
                if shared
                else "請確認是否為個人帳；若仍為群組帳應排程汰換。"
            ),
        ]
        if telnet:
            s2.append(
                "傳輸風險：出現 Telnet（TCP 23）明文管理通道，帳密與指令可被竊聽，屬高風險，應立即禁用。"
            )
        if ssh:
            s2.append(
                "正向訊號：有 SSH（TCP 22）登入；仍須限制來源網段、禁弱密碼並啟用 AAA。"
            )
        if login_fail:
            s2.append(
                f"失敗登入 {len(login_fail)} 筆：可能為密碼錯誤、探測或憑證噴灑，需與鎖定事件關聯。"
            )
        if any(e["mnemonic"] == "TTY_EXPIRE_TIMER" for e in logout):
            s2.append(
                "正向控制：出現閒置逾時（TTY_EXPIRE），代表有 session timeout，有助降低無人看管終端風險。"
            )
        s3 = [
            "立即：transport input ssh（禁用 Telnet／rlogin）；管理面 ACL／VRF 只放行跳板與網管網段。",
            "帳號：導入 RADIUS／TACACS+ 個人帳與指令授權；廢止共用 cisco／enable 密碼共用。",
            "監控：對非白名單來源 LOGIN_SUCCESS、短時間 LOGIN_FAILED 暴增做 SIEM 告警。",
            "稽核：登入／登出與變更單對帳；特權操作保留完整 tty 紀錄至少符合保留政策。",
            "加固：SSHv2、足夠金鑰長度、exec-timeout、關閉未用 HTTP／過時管理服務。",
        ]
        return _build_report_cards(s1, s2, s3)

    if malware:
        code_list = ", ".join(sorted({e["code"] for e in malware})[:6])
        samples = _sample_raw_lines(malware, 3)
        s1 = [
            f"控制項「{title}」：偵測到惡意軟體／IPS／埠安全相關 syslog 共 {len(malware)} 筆。",
            f"訊息類型：{code_list}。設備：{_devices_from_events(malware)}。",
            "應以資安事件視之，不是一般維運雜訊；禁止改寫成無關的組態增刪劇本。",
        ]
        if samples:
            s1.append("代表性原文：" + " ｜ ".join(samples[:2]))
        if metric:
            s1.append(f"控制項量化：{metric[:200]}")
        s2 = [
            "合規對應：ISO 27001 A.8.7（惡意軟體防護）及相關偵測控制。",
            "衝擊：可能造成無塵室／製程網可用性下降，或成為橫向移動與勒索入口。",
            "PORT_SECURITY／PSECURE：常代表未授權 MAC 或埠安全違規，需確認是否為設備更換未更新綁定。",
            "IPS／MALWARE：若為封鎖成功仍需追查來源與殘留；若為偵測未阻斷則風險更高。",
            "嚴重度：建議至少列 review；計數偏高或含 Critical severity 應升級事件單。",
        ]
        s3 = [
            "應變：隔離相關埠／VLAN／主機，保全前後 syslog 與埠狀態截圖。",
            "排查：show port-security、比對資產 MAC、確認是否維護誤插或未知裝置。",
            "防護：強化 802.1X／MAB、USB 管制、終端白名單；更新 IPS／AV 簽章。",
            "流程：開立資安事件、評估是否需通報與還原；完成後更新基線與教訓學習。",
            "監控：對 MALWARE／IPS／PSECURE 設即時告警，避免只在匯出報告時才發現。",
        ]
        return _build_report_cards(s1, s2, s3)

    if snmp:
        code_list = ", ".join(sorted({e["code"] for e in snmp})[:6])
        authfail = [e for e in snmp if "AUTHFAIL" in e["code"] or "COMMUNITY" in e["code"]]
        crypto = [
            e for e in snmp
            if re.search(r"CRYPTO|PKI|IKE|CERT", e["code"], re.I)
        ]
        samples = _sample_raw_lines(snmp, 3)
        s1 = [
            f"控制項「{title}」：出現 SNMP／加密／憑證相關事件 {len(snmp)} 筆（{code_list}）。",
            f"其中認證失敗／community 類 {len(authfail)} 筆；CRYPTO／PKI／IKE 類 {len(crypto)} 筆。",
            "屬傳輸與密碼學控制範圍（A.8.24），不是介面抖動或配方變更。",
        ]
        if samples:
            s1.append("代表性原文：" + " ｜ ".join(samples[:2]))
        if metric:
            s1.append(f"控制項量化：{metric[:200]}")
        s2 = [
            "合規對應：ISO 27001 A.8.24 密碼學與傳輸安全。",
            "SNMPv1/v2c 明文 community 易被竊聽與濫用；AUTHFAIL 可能是錯誤設定或未授權輪詢／探測。",
            "憑證無效、IKE 失敗或金鑰過短會削弱管理面與隧道完整性。",
            "若計數偏高但無對應 NMS 維護窗口，應懷疑掃描或設定漂移。",
        ]
        s3 = [
            "升級／強制 SNMPv3（authPriv）；淘汰明文 community 並定期輪替。",
            "以 ACL 限制僅信任 NMS／監控主機可存取 UDP 161／162。",
            "檢查 PKI 憑證效期與信任鏈；修正 IKE／IPSec 提案與時鐘同步（NTP）。",
            "監控 AUTHFAIL／CERTIFICATE 告警；異常來源納入封鎖與追查。",
            "文件化：SNMP 帳密／金鑰納入機密管理，避免寫在共用帳文件。",
        ]
        return _build_report_cards(s1, s2, s3)

    if patch:
        code_list = ", ".join(sorted({e["code"] for e in patch})[:6])
        samples = _sample_raw_lines(patch, 3)
        s1 = [
            f"控制項「{title}」：出現開機／映像／安裝／平台類事件 {len(patch)} 筆（{code_list}）。",
            f"設備：{_devices_from_events(patch)}。此類事件對應技術弱點與變更窗口管理（A.8.8）。",
        ]
        if samples:
            s1.append("代表性原文：" + " ｜ ".join(samples[:2]))
        if metric:
            s1.append(f"控制項量化：{metric[:200]}")
        s2 = [
            "合規對應：ISO 27001 A.8.8 技術弱點管理。",
            "未經驗證的映像、延遲修補或非窗口重載，會擴大已知 CVE 暴露面並影響產線可用性。",
            "需區分：排程升級（有變更單）vs 非預期重載／安裝失敗。",
        ]
        s3 = [
            "僅安裝 Cisco 簽章映像；升級前做 MD5／SHA 驗證與實驗室驗證。",
            "維護窗口執行；保留前一版可回退；更新弱點清冊與 PSIRT 追蹤。",
            "重載／INSTALL 完成後做健康檢查（介面、BGP／OSPF、關鍵 VLAN）。",
            "禁止在無變更單情況下於生產時段做映像操作。",
        ]
        return _build_report_cards(s1, s2, s3)

    if supplier:
        code_list = ", ".join(sorted({e["code"] for e in supplier})[:6])
        samples = _sample_raw_lines(supplier, 3)
        s1 = [
            f"控制項「{title}」：出現鄰近發現／PnP／遠端 logging 等外部通道事件 "
            f"{len(supplier)} 筆（{code_list}）。",
            f"設備：{_devices_from_events(supplier)}。對應供應商與外部連線治理（A.5.19）。",
        ]
        if samples:
            s1.append("代表性原文：" + " ｜ ".join(samples[:2]))
        if metric:
            s1.append(f"控制項量化：{metric[:200]}")
        s2 = [
            "合規對應：ISO 27001 A.5.19 供應商關係／外部連線。",
            "CDP／LLDP 可能洩漏拓樸；PnP 不當啟用可能引入未授權佈建。",
            "遠端 logging 若送往非信任 collector，有日誌外洩或遭竄改風險。",
            "計數偏高不代表立即遭駭，但代表外部可見性／連線面需治理。",
        ]
        s3 = [
            "非必要關閉 CDP／LLDP 或限制於管理 VLAN；審核 PnP profile。",
            "遠端 log 僅送信任主機，優先 TLS／可靠傳輸，並做來源驗證。",
            "供應商遠端維運走核准跳板、時限帳號與全程錄影／指令稽核。",
            "定期盤點外部連線與合約資安條款，移除無用通道。",
        ]
        return _build_report_cards(s1, s2, s3)

    if config or power:
        focus = config or power
        code_list = ", ".join(sorted({e["code"] for e in focus})[:6])
        samples = _sample_raw_lines(focus, 3)
        s1 = [
            f"控制項「{title}」：出現組態／供電／生成樹／errdisable 類事件 "
            f"{len(focus)} 筆（{code_list}）。",
            "僅依實際訊息碼解讀；不可虛構 MFG 控制器或設定檔增刪細節。",
        ]
        if samples:
            s1.append("代表性原文：" + " ｜ ".join(samples[:2]))
        if metric:
            s1.append(f"控制項量化：{metric[:200]}")
        s2 = [
            "合規對應：A.8.19 組態變更與營運事件稽核。",
            "CONFIG_I 無變更單可能破壞安全基線；ILPOWER／errdisable 則偏可用性與供電。",
            "需把「變更」與「故障抖動」分開：前者看審批，後者看實體與 PoE。",
        ]
        s3 = [
            "組態：強制變更單、archive 差異比對、限制可寫入管理帳。",
            "供電／errdisable：查 show power inline、errdisable 原因並修復後再 recovery。",
            "所有變更與復原留存證據，納入定期合規複核。",
        ]
        return _build_report_cards(s1, s2, s3)

    # 其餘可辨識 Cisco 事件
    sample_bodies = "；".join(e["body"][:100] for e in events[:4])
    samples = _sample_raw_lines(events, 3)
    s1 = [
        f"控制項「{title}」：辨識到 {len(events)} 筆 Cisco syslog（{', '.join(codes[:6])}）。",
        f"設備：{_devices_from_events(events)}。內容摘要：{sample_bodies}",
    ]
    if samples:
        s1.append("代表性原文：" + " ｜ ".join(samples[:2]))
    if metric:
        s1.append(f"控制項量化：{metric[:200]}")
    s2 = [
        "請嚴格依 facility／mnemonic 與正文解讀；禁止虛構未出現的設備名、配方或攻擊劇本。",
        "依事件類型評估對可用性、存取控制、傳輸安全或惡意防護的影響。",
        "若同一訊息碼反覆出現，應提高優先序並關聯變更窗口與資產清單。",
    ]
    s3 = [
        "對照變更／維運窗口；重複異常設告警並保全本段原始日誌。",
        "必要時隔離相關埠位，使用 show logging／show interface 補充證據。",
        "完成排查後更新監控基線與事件結案紀錄（CAPA）。",
    ]
    return _build_report_cards(s1, s2, s3)


def _report_hallucinates_against_log(reply: str, log_text: str) -> bool:
    """模型是否寫出「證據裡沒有」的關鍵幻覺（log／監控／對話皆可用）。"""
    r = reply or ""
    evidence = log_text or ""
    # 佔位符假 log（訓練殘留模板）
    if re.search(
        r"\[HOSTNAME\]|\[IP_ADDRESS\]|\[PORT\]|\[SYSLOG_LEVEL\]|\[SYSLOG_MSG\]|"
        r"SYSLOG:\s*\[ID\s*\d+\]|\[ID\s*1000\]",
        r,
        re.I,
    ):
        return True
    # 經典幻覺：MFG 控制器／增刪設定檔／P01 機臺
    if re.search(r"MFG\s*\d+|MFG01|P01\s*機臺|P01機臺", r, re.I) and not re.search(
        r"MFG|P01", evidence, re.I
    ):
        return True
    if re.search(r"新增(?:新)?設定檔|刪除舊設定檔|控制器進行變更組態", r) and not re.search(
        r"CONFIG_I|Configured from|設定檔", evidence, re.I
    ):
        return True
    # 異場域／訓練污染
    if _FOREIGN_SITE_RE.search(r) and not _FOREIGN_SITE_RE.search(evidence):
        return True
    # 虛構常見訓練 IP（證據沒有卻出現）
    for ip in re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", r):
        if ip.startswith("127.") or ip.startswith("0."):
            continue
        if ip not in evidence and ip in ("192.168.1.100", "192.168.0.1", "10.0.0.1"):
            return True
    # 虛構時間戳假事件
    if re.search(r"\[INFO\]\s*A\.\d+|本次日誌：\s*20\d{2}", r) and not re.search(
        r"%[A-Z0-9_-]+-\d+-[A-Z0-9_]+", evidence, re.I
    ):
        if re.search(r"HOSTNAME|SYSLOG:\s*\[ID|密碼學與網絡傳輸安全校驗檢查結果", r):
            return True
    for dm in re.finditer(
        r"20\d{2}[-/年]\d{1,2}[-/月]\d{1,2}",
        r,
    ):
        token = re.sub(r"[年月]", "-", dm.group(0)).replace("/", "-")
        if token[:10] not in evidence and dm.group(0) not in evidence:
            if not re.search(r"20\d{2}", evidence) and re.search(
                r"SYSLOG|HOSTNAME|\[INFO\]", r, re.I
            ):
                return True
    # UPDOWN／LOGIN 證據卻講重放／閥門
    if re.search(r"UPDOWN|SEC_LOGIN|LOGIN_SUCCESS", evidence, re.I) and re.search(
        r"重放攻擊|Replay Attack|開啟.*閥門|Modbus\s*寫入|配方下載", r, re.I
    ):
        return True
    # 證據只有監控摘要、回覆卻捏造具體 syslog 行
    if evidence and not re.search(r"%[A-Z0-9_-]+-\d+-[A-Z0-9_]+", evidence, re.I):
        if re.search(
            r"%[A-Z0-9_-]+-\d+-[A-Z0-9_]+|\[INFO\]\s*A\.\d+",
            r,
            re.I,
        ) and re.search(r"本次日誌|SYSLOG:\s*\[", r):
            return True
    # 訓練殘留：假「報告工單／執行中／Admin／2023」狀態頁
    if re.search(
        r"事件類型\s*[:：]\s*資安報告|生成報告狀態\s*[:：]|使用者名稱\s*[:：]\s*Admin|"
        r"預計\s*\d+\s*小時內完成|事件狀態\s*[:：]\s*(進行中|執行中)|"
        r"報告內容\s*[:：].*系統名稱",
        r,
        re.I,
    ):
        return True
    return False


def _chat_evidence_blob(user_message="", ot_context="", rag_context="") -> str:
    return "\n".join(
        x for x in (user_message or "", ot_context or "", rag_context or "") if x
    )


def _filter_rag_for_chat(rag_context: str, user_message: str) -> str:
    """對話用 RAG：去掉易引發幻覺／跑題的片段。"""
    if not rag_context:
        return ""
    um = user_message or ""
    wants_log = bool(
        re.search(
            r"%[A-Z0-9_-]+-\d+|syslog|SEC_LOGIN|LOGIN_|Cisco|日誌|訊息碼",
            um,
            re.I,
        )
    )
    wants_27001 = _wants_iso27001_topic(um)
    keep = []
    for block in re.split(r"\n\s*\n", rag_context):
        b = block.strip()
        if not b:
            continue
        if re.search(r"MFG01|HOSTNAME|重放攻擊.*LOGIN|LOGIN.*重放攻擊", b, re.I):
            continue
        if _FOREIGN_SITE_RE.search(b) and not re.search(
            r"半導體|Cisco|SEC_LOGIN|UPDOWN|SNMP", b, re.I
        ):
            continue
        # 使用者問登入時，丟掉純重放攻擊教材
        if re.search(r"LOGIN|登入|SSH|Telnet", um, re.I) and re.search(
            r"重放攻擊|Replay Attack|閥門", b, re.I
        ):
            continue
        # 純知識／27001：丟掉三卡報告體／訓練對話殘留（最易把回答帶歪）
        if (wants_27001 or not wants_log) and re.search(
            r"地端 LLM 智慧合規|事件經過摘要|不合規／風險分析|具體修補建議|"
            r"使用者[：:]|assistant[：:]|【報告結束】|一、事件|二、不合規",
            b,
            re.I,
        ):
            continue
        # 知識問答時丟掉整段假設備／假時間戳劇本
        if not wants_log and re.search(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}|C9300-\d+|GigabitEthernet\d|"
            r"Admin\d{2,}|工單[#＃]?\d{4,}",
            b,
            re.I,
        ):
            continue
        keep.append(b)
    # 對話最多留 2 塊，避免小模型被長教材帶跑
    return "\n\n".join(keep[:2])


def _query_needs_log_rag(query: str) -> bool:
    """是否為「日誌／訊息碼解讀」類（才適合灌 OT log 教材）。"""
    t = query or ""
    return bool(
        re.search(
            r"%[A-Z0-9_-]+-\d+-[A-Z0-9_]+|"
            r"\b(SEC_LOGIN|LOGIN_FAILED|LOGIN_SUCCESS|CONFIG_I|UPDOWN|"
            r"RADIUS|TACACS|SNMP-|CRYPTO-|ILPOWER)\b|"
            r"syslog|cisco\s*log|交換器.*日誌|日誌.*分析|訊息碼",
            t,
            re.I,
        )
    )


def wants_remediation_steps(user_message: str) -> bool:
    """是否在問修補／修復／改善步驟（聊天常見，不宜丟給不穩的 LLM）。"""
    t = (user_message or "").strip()
    if not t:
        return False
    return bool(
        re.search(
            r"修補步驟|修復步驟|改善步驟|怎麼修|如何修|怎麼改|如何改善|"
            r"修補建議|防護建議|處置步驟|應變步驟|說明修補|講解修補|"
            r"patch(ing)?\s*steps|remediat",
            t,
            re.I,
        )
    )


def wants_hardening_howto(user_message: str) -> bool:
    """是否在問弱協議停用／管理面加固等可執行設定步驟。"""
    t = (user_message or "").strip()
    if not t:
        return False
    return bool(
        re.search(
            r"(停用|禁用|關閉|如何|怎麼|怎麽|要怎麼|怎麼執行|如何執行)"
            r".{0,16}(telnet|ssh|https|snmpv?1|明文|弱協議)|"
            r"(telnet|弱協議|明文登入).{0,16}(停用|禁用|關閉|怎麼|如何|步驟|執行)|"
            r"transport\s*input\s*ssh|"
            r"(管理面|交換器|cisco|catalyst).{0,20}(加固|hardening|最小加固)",
            t,
            re.I,
        )
    )


def build_hardening_howto_reply(user_message: str = "") -> str:
    """Cisco 管理面加固／停用 Telnet 等 grounded 步驟（固定結論／說明／建議格式）。"""
    t = (user_message or "").strip()
    focus_telnet = bool(re.search(r"telnet|明文|transport\s*input", t, re.I)) or not re.search(
        r"snmp|https|aaa|tacacs", t, re.I
    )
    details = [
        "VTY 設定 `transport input ssh`（拒絕 Telnet／TCP 23；勿再用 telnet 或 all）",
        "先完成：`ip domain-name` → `crypto key generate rsa modulus 2048` → `ip ssh version 2`",
        "line vty 0 15：`login local`（或 AAA）＋`exec-timeout 10 0`",
        "帳號改個人帳，對齊 RADIUS／TACACS+；特權用 `enable secret`",
        "建議以 ACL／`access-class` 限制網管／跳板來源",
        "範例指令：",
        "```",
        "configure terminal",
        "ip ssh version 2",
        "line vty 0 15",
        " transport input ssh",
        "exit",
        "write memory",
        "```",
    ]
    if not focus_telnet:
        details.extend([
            "HTTP：`no ip http server`，改 HTTPS（`ip http secure-server`）",
            "SNMP：關閉弱 community，改 SNMPv3",
        ])
    return format_fixed_chat_reply(
        "應停用 Telnet 明文管理，僅允許 SSHv2，並對齊 ISO 27001 存取控制（A.7.4／A.11.2）。",
        details,
        [
            "用 `show ip ssh`、`show line vty 0 15` 確認 transport 只有 ssh",
            "SSH 測通後再中斷舊 Telnet session，並監看 `%SEC_LOGIN`",
            "變更前走變更管理，確認不影響既有網管／自動化連線",
        ],
    )


def wants_data_counts(user_message: str) -> bool:
    """是否在問資料／日誌筆數、事件量等可量化數字。"""
    t = (user_message or "").strip()
    if not t:
        return False
    return bool(
        re.search(
            r"資料筆數|日誌筆數|事件筆數|有幾筆|多少筆|幾筆資料|筆數|"
            r"事件量|資料量|總筆數|計算.*筆|統計.*筆|多少(條|行|筆)|"
            r"(幾|多少).{0,6}(檔|設備)|num_files|total_lines|"
            r"count\s*(是多少|多少)|事件數",
            t,
            re.I,
        )
    )


def wants_ot_status_summary(user_message: str) -> bool:
    """合規／監控現況、控制項狀態：應直接回真實計數，勿經 LLM。"""
    t = (user_message or "").strip()
    if not t or is_casual_chat(t) or is_off_topic_chat(t):
        return False
    if wants_visual(t) or wants_hardening_howto(t):
        return False
    return bool(
        re.search(
            r"合規現況|監控現況|稽核現況|目前現況|現況如何|現在狀況|"
            r"各控制項|控制項.*(狀態|計數|數字|現況)|六大控制|"
            r"metrics|監控計數|戰情.*(現況|狀態)|"
            r"(目前|現在).{0,6}(合規|監控|狀態|計數)",
            t,
            re.I,
        )
    )


_FAKE_METRIC_KEY_RE = re.compile(
    r"\b(?:"
    r"network_monitor|intrusion_detection|firewall_log|security_event_log|"
    r"vulnerability_scanning|log_analysis|incident_response|compliance_check|"
    r"asset_management|threat_detection|siem_events|ids_alerts|ups_monitor|"
    r"endpoint_protection|patch_status|user_access_log"
    r")\b",
    re.I,
)
_FAKE_SEQ_COUNT_RE = re.compile(
    r"\b(?:12345|23456|34567|45678|56789|67890|78901|89012|90123|01234)\b"
)


def _looks_like_fake_metrics_reply(text: str) -> bool:
    """偵測模型捏造的假控制項／連續假數字（訓練殘留）。"""
    t = text or ""
    if not t.strip():
        return False
    if _FAKE_METRIC_KEY_RE.search(t):
        return True
    if _FAKE_SEQ_COUNT_RE.search(t) and re.search(r"status\s*=", t, re.I):
        return True
    if re.search(r"\bUPDTS\b", t) and re.search(r"status\s*=", t, re.I):
        return True
    unknown = 0
    for m in re.finditer(
        r"^\s*[\*\-•]\s*`?([a-z][a-z0-9_]{2,40})`?\s*:\s*\d+",
        t,
        re.I | re.M,
    ):
        if m.group(1).lower() not in CONTROL_TITLES:
            unknown += 1
    return unknown >= 3


def build_ot_data_counts_reply() -> str | None:
    """依目前 OT 掃描結果回覆真實筆數（不經 LLM）。"""
    import json as _json

    data = get_ot_monitor_data()
    if not isinstance(data, dict) or data.get("error"):
        return None

    metrics = data.get("metrics") or {}
    files = []
    num_devices = 0
    num_files = 0
    try:
        summary = _json.loads(data.get("all_logs_content") or "{}")
        files = summary.get("files") or []
        num_devices = int(summary.get("num_devices") or 0)
        num_files = int(summary.get("num_files") or len(files) or 0)
    except Exception:
        files = []

    total_lines = sum(int(f.get("total_lines") or 0) for f in files)
    event_lines = sum(int(f.get("event_lines") or 0) for f in files)
    if not num_files:
        num_files = len(files)

    details = [
        f"**日誌檔數**：{num_files} 個 TXT",
        f"**設備數**：{num_devices or '—'}",
        f"**原始總行數**：{total_lines:,} 行",
        f"**可分類事件列**：{event_lines:,} 筆",
    ]
    control_sum = 0
    for key, title in CONTROL_TITLES.items():
        m = metrics.get(key) or {}
        c = int(m.get("count") or 0)
        control_sum += c
        st = m.get("status", "n/a")
        details.append(f"{title}：**{c:,}**（status={st}）")
    details.append(
        f"**六控制項計數合計**：{control_sum:,}"
        "（同一行可能只歸一類；合計可能與可分類事件列略有差異）"
    )
    for f in sorted(files, key=lambda x: -int(x.get("total_lines") or 0))[:8]:
        details.append(
            f"`{f.get('file')}`（{f.get('device') or 'n/a'}）："
            f"總行 {int(f.get('total_lines') or 0):,}／"
            f"事件 {int(f.get('event_lines') or 0):,}"
        )
    if len(files) > 8:
        details.append(f"…另有 {len(files) - 8} 個檔案未列出")

    return format_fixed_chat_reply(
        "以下為目前 `ot/` 日誌掃描的真實筆數（非估算）。",
        details,
        [
            "若要看風險解讀可問合規相關問題或「修補步驟」",
            "若要分析單一事件請直接貼 Cisco syslog",
        ],
    )


def _grounded_draft_prompt_block(
    draft: str,
    *,
    purpose: str = "結構化診斷草稿",
    max_chars: int = 1400,
) -> str:
    """將預先生成的事實底稿格式化成 prompt 區塊（類 RAG 注入）。"""
    trimmed = (draft or "").strip()
    if not trimmed:
        return ""
    if len(trimmed) > max_chars:
        trimmed = trimmed[:max_chars] + "\n...(底稿已截斷)..."
    return (
        f"\n【{purpose}（系統預先生成｜必須依此改寫）】\n"
        "下列為依真實日誌／監控指標產出的結構化事實底稿；"
        "請用你自己的語句重新組織，但 count、status、訊息碼、設備名等事實不可改、不可增刪。\n"
        "禁止整段照抄；禁止與底稿矛盾（例如底稿寫 LOGIN_SUCCESS 勿改寫成登入失敗或暴力破解）。\n"
        f"{trimmed}\n"
    )


def _build_audit_grounded_draft(
    *,
    log_text: str,
    control_key: str | None,
    control_title: str,
    metric_summary: str | None,
    rag_context: str | None,
    cisco_events: list,
) -> str:
    """稽核報告：先產結構化三卡事實底稿，再供 LLM 改寫。"""
    if ENABLE_CISCO_GROUNDED_REPORT and cisco_events:
        draft = build_cisco_log_grounded_report(
            log_text,
            control_title=control_title,
            metric_summary=metric_summary,
        )
        if draft and _cjk_count(draft) >= 36:
            return draft
    draft = build_factual_report(
        control_title=control_title,
        metric_summary=metric_summary,
        log_text=log_text,
        rag_context=rag_context,
    )
    if draft and _cjk_count(draft) >= 36:
        return draft
    return build_metric_only_report(
        control_title=control_title,
        metric_summary=metric_summary,
        control_key=control_key,
    )


def build_ot_compliance_status_grounded(user_message: str = "") -> str | None:
    """合規現況：結構化事實底稿（注入 LLM，非直接回覆使用者）。"""
    detailed = bool(re.search(r"詳細|深入|細部|完整|分析", user_message or "", re.I))
    if detailed:
        return build_ot_situation_grounded_reply()

    data = get_ot_monitor_data()
    if not isinstance(data, dict) or data.get("error"):
        return None
    metrics = data.get("metrics") or {}

    fails: list[str] = []
    reviews: list[str] = []
    passes: list[str] = []
    for key, title in CONTROL_TITLES.items():
        m = metrics.get(key) or {}
        st = str(m.get("status") or "n/a").lower()
        count = int(m.get("count") or 0)
        annex = CONTROL_LABEL_ALIAS.get(key, key)
        line = f"- {title.split('（')[0].strip()}（{annex}）：{count:,} 筆，status={st}"
        if st == "fail":
            fails.append(line)
        elif st == "review":
            reviews.append(line)
        else:
            passes.append(line)

    risks: list[str] = []
    if (metrics.get("access_control") or {}).get("status") == "fail":
        risks.append(
            "- A.5.15 存取控制 status=fail：驗證事件量偏高需稽核；"
            "LOGIN_SUCCESS 是管理登入紀錄，勿解讀成登入失敗或暴力破解。"
        )
    if (metrics.get("malware_defense") or {}).get("count", 0) > 0:
        risks.append("- A.8.7 惡意防護有命中計數：需依 syslog 隔離追查。")
    if (metrics.get("recipe_audit") or {}).get("status") == "review":
        risks.append("- A.8.19 組態／介面 status=review：可能含 UPDOWN 抖動，需排查。")
    if not risks:
        risks.append("- 目前無單一控制項呈 fail；仍建議抽樣複核原文 syslog。")

    steps = [
        "- 優先處理 fail 控制項：保全樣本 syslog 並開事件單。",
        "- 可問「修補步驟」取得具體作法；正式報告請至監控戰情室匯出 PDF／TXT。",
    ]

    parts = [
        "## 合規現況結構化底稿",
        "",
        "### 一、監控計數摘要",
        f"fail {len(fails)} 項、review {len(reviews)} 項、pass {len(passes)} 項。",
        "",
    ]
    if fails:
        parts.append("**fail（需優先）**")
        parts.extend(fails)
        parts.append("")
    if reviews:
        parts.append("**review（持續關注）**")
        parts.extend(reviews)
        parts.append("")
    if passes:
        parts.append("**pass**")
        parts.extend(passes)
        parts.append("")
    parts.extend(["### 二、風險重點", ""])
    parts.extend(risks)
    parts.extend(["", "### 三、建議下一步", ""])
    parts.extend(steps)
    return "\n".join(parts)


def build_ot_compliance_status_reply(user_message: str = "") -> str | None:
    """合規現況：LLM 失敗時的格式化 fallback。"""
    grounded = build_ot_compliance_status_grounded(user_message)
    if not grounded:
        return None
    bullets = [
        ln.strip()
        for ln in grounded.splitlines()
        if ln.strip() and not re.match(r"^#+\s", ln.strip())
    ][:14]
    return format_fixed_chat_reply(
        "以下為依目前 OT 監控產出的合規現況（系統事實底稿）。",
        bullets,
        ["可問「修補步驟」", "正式報告請至監控戰情室匯出 PDF／TXT"],
    )


def _extract_requested_item_count(user_message: str) -> int:
    """解析「給我 10 個修補建議」這類數量要求。"""
    t = (user_message or "").strip()
    if not t:
        return 0
    for pat in (
        r"(\d+)\s*(?:個|项|条|項|條)?\s*(?:修補|建議|步驟|做法|措施)",
        r"(?:給我|列出|提供|要)\s*(\d+)",
        r"(\d+)\s*(?:patch|remediation|recommendations?)",
    ):
        m = re.search(pat, t, re.I)
        if m:
            return min(max(int(m.group(1)), 1), 20)
    return 0


def build_ot_remediation_steps_reply(user_message: str = "") -> str | None:
    """依監控 fail／review 控制項組出可執行修補步驟（不經 LLM）。"""
    data = get_ot_monitor_data()
    if not isinstance(data, dict) or data.get("error"):
        return None
    metrics = data.get("metrics") or {}

    steps_by_key = {
        "access_control": [
            "盤點共用帳與特權帳，改個人帳＋最小權限（A.7.4／A.11.2）。",
            "停用 Telnet／明文登入，強制 SSH／HTTPS，並對齊 RADIUS／TACACS+。",
            "對失敗登入暴增來源做來源 IP／時段對帳，必要時暫時封鎖。",
            "完成後抽樣複核 SEC_LOGIN／LOGIN 日誌，確認異常下降。",
        ],
        "malware_defense": [
            "保全 MALWARE／IPS／PSECURE／USB 命中原文，開立事件單（A.8.7）。",
            "隔離可疑埠／主機，更新簽章與 MAC／802.1X／port-security。",
            "禁用未授權 USB，複核交換器政策是否一致下發。",
            "清除威脅後做教訓學習與基線更新。",
        ],
        "recipe_audit": [
            "對 UPDOWN／組態變更高發埠做實體連線與 flop 排查（A.8.19）。",
            "核對變更是否有變更單；無單變更先凍結並回溯。",
            "必要時啟用 errdisable recovery 與介面告警門檻。",
            "穩定後再放行生產變更窗口。",
        ],
        "sec_gem_log": [
            "改 SNMPv3、汰換預設 community，限制 NMS 來源（A.8.24）。",
            "檢查 syslog／SNMP trap 傳輸是否加密或走管理 VRF。",
            "複核憑證／金鑰效期與錯誤 trap 來源。",
            "抽樣確認敏感傳輸不再走明文。",
        ],
        "patch_management": [
            "盤點交換器／防火牆／工控主機版本與已知 CVE（A.8.8）。",
            "在維護窗分批升級，先測後正式，保留回退映像。",
            "升級後驗證核心服務與日誌收集正常。",
            "更新組態基線與修補紀錄。",
        ],
        "supplier_security": [
            "清點供應商／維運遠端管道與帳號時效（A.5.19）。",
            "要求供應商帳號 MFA／限期帳，並保留操作日誌。",
            "對 CDP／LLDP／外部連線異常做來源與合約對帳。",
            "合約條款補齊資安責任與事件通報 SLA。",
        ],
    }

    priority = []
    for key, title in CONTROL_TITLES.items():
        m = metrics.get(key) or {}
        status = (m.get("status") or "").lower()
        count = m.get("count", 0)
        if status in ("fail", "review") or (
            key == "malware_defense" and count > 0
        ):
            priority.append((key, title, status or "n/a", count))

    if not priority:
        return format_fixed_chat_reply(
            "目前各控制項未呈現明確 fail／review，建議維持例行維運。",
            [
                "定期掃描 ot 日誌並抽樣人工覆核",
                "維持 AAA、SNMPv3、USB／埠安全與變更管制",
                "步驟僅依監控計數／狀態，未虛構單一設備事件",
            ],
            [
                "若要針對某控制項，請說明控制項名稱或貼上 syslog",
            ],
        )

    requested = _extract_requested_item_count(user_message)
    if requested:
        flat: list[str] = []
        for key, title, status, count in priority:
            for s in steps_by_key.get(
                key,
                [
                    "保全相關日誌樣本並對照變更單。",
                    "依 ISO 控制項完成修正後複測。",
                ],
            ):
                flat.append(f"{title}（status={status}，count={count}）：{s}")
        if len(flat) < requested:
            for key, title in CONTROL_TITLES.items():
                if any(key == p[0] for p in priority):
                    continue
                m = metrics.get(key) or {}
                status = (m.get("status") or "n/a").lower()
                count = m.get("count", 0)
                for s in steps_by_key.get(key, [])[:2]:
                    flat.append(f"{title}（status={status}，count={count}）：{s}")
                if len(flat) >= requested:
                    break
        numbered = [f"{i}. {s}" for i, s in enumerate(flat[:requested], start=1)]
        return format_fixed_chat_reply(
            f"依目前 OT 監控狀態整理 {len(numbered)} 項修補建議（僅依計數／狀態，未虛構單一設備事件）。",
            numbered,
            [
                "請依 fail／review 控制項優先處理，完成後複核對應日誌",
                "若要針對某一筆 Cisco syslog 逐步說明，請直接貼上原文",
            ],
        )

    details = []
    for key, title, status, count in priority[:4]:
        details.append(f"**{title}**（status={status}，count={count}）")
        for s in steps_by_key.get(
            key,
            [
                "保全相關日誌樣本並對照變更單。",
                "依 ISO 控制項完成修正後複測。",
            ],
        ):
            details.append(s)
    return format_fixed_chat_reply(
        "依目前 OT 監控狀態整理修補步驟（僅依計數／狀態，未虛構單一設備事件）。",
        details,
        [
            "請依 fail／review 控制項優先處理，完成後複核對應日誌",
            "若要針對某一筆 Cisco syslog 逐步說明，請直接貼上原文",
        ],
    )


def build_ot_visual_brief_reply(user_message: str = "") -> str:
    """圖表請求：短文說明＋交由 ensure_visual_reply 補 chart（不經 LLM／不塞修補長文）。"""
    data = get_ot_monitor_data()
    metrics = (data or {}).get("metrics") or {} if isinstance(data, dict) else {}
    bits = []
    for key, title in CONTROL_TITLES.items():
        m = metrics.get(key) or {}
        bits.append(f"{CONTROL_LABEL_ALIAS.get(key, key)}={m.get('count', 0)}")
    summary = "、".join(bits[:6]) if bits else "尚無監控計數"
    chart_hint = "折線圖" if re.search(r"折線|line", user_message or "", re.I) else "圖表"
    return format_fixed_chat_reply(
        f"已依目前 OT 各控制項事件量準備{chart_hint}。",
        [
            f"監控計數摘要：{summary}",
            "下方圖表資料來自即時監控計數，未虛構設備事件",
        ],
        ["可再問修補步驟，或貼 syslog 做單筆分析"],
    )


def build_ot_situation_grounded_reply() -> str | None:
    """依目前監控 bundle 組出「現況診斷」，不呼叫 LLM。"""
    data = get_ot_monitor_data()
    if not isinstance(data, dict) or data.get("error"):
        return None
    bundles = data.get("control_bundles") or {}
    metrics = data.get("metrics") or {}
    parts = [
        "## 地端 LLM 智慧合規診斷報告",
        "",
        "## 一、事件經過摘要",
        "",
        "以下依各控制項真實樣本／計數彙整現況（未虛構 HOSTNAME、MFG 或假 syslog）：",
    ]
    highlights = []
    for key, title in CONTROL_TITLES.items():
        b = bundles.get(key) or {}
        m = metrics.get(key) or {}
        log = b.get("log") or ""
        count = m.get("count", b.get("event_count", 0))
        status = m.get("status", "n/a")
        events = _extract_cisco_events(log)
        if events:
            # 盡量多樣化訊息碼（避免只剩 LOGIN_SUCCESS）
            codes = []
            for e in events:
                c = e.get("code") or ""
                if c and c not in codes:
                    codes.append(c)
                if len(codes) >= 3:
                    break
            status_note = ""
            if key == "access_control" and status == "fail":
                status_note = "（fail＝驗證事件量偏高需稽核，≠登入失敗）"
            highlights.append(
                f"- **{title}**：status={status}{status_note}，count={count}；"
                f"樣本訊息碼 {', '.join(codes[:3])}"
            )
        else:
            highlights.append(
                f"- **{title}**：status={status}，count={count}；"
                f"{'尚無抽到原文樣本（已計數）' if count else '無命中事件'}"
            )
    parts.extend(highlights[:8])

    parts.extend(["", "## 二、不合規／風險分析", ""])
    risks = []
    if (metrics.get("access_control") or {}).get("status") in ("fail", "review"):
        risks.append(
            "- 存取／驗證事件量偏高（含 LOGIN_SUCCESS）："
            "需核對帳號合法性、共用帳、Telnet 與來源白名單，"
            "不是把成功登入解讀成攻擊失敗（A.7.4）。"
        )
    if (metrics.get("malware_defense") or {}).get("count", 0) > 0:
        risks.append(
            "- 出現惡意／入侵相關計數：應隔離追查，勿寫成「無風險」（A.8.7）。"
        )
    if (metrics.get("recipe_audit") or {}).get("status") == "review":
        risks.append(
            "- 組態／介面事件偏多：可能含 UPDOWN 抖動，先做實體連線排查（A.8.19）。"
        )
    if (metrics.get("sec_gem_log") or {}).get("status") == "review":
        risks.append(
            "- 傳輸／SNMP／加密事件需複核 community、憑證與 NMS 來源（A.8.24）。"
        )
    if (metrics.get("patch_management") or {}).get("status") == "review":
        risks.append(
            "- 弱點／供電／FRU 類事件偏多：安排維護窗盤點版本與硬體健康（A.8.8）。"
        )
    if (metrics.get("supplier_security") or {}).get("status") == "review":
        risks.append(
            "- 供應商關係相關計數偏高：核對遠端維運帳號時效與連線來源（A.5.19）。"
        )
    if not risks:
        risks.append("- 目前無單一控制項呈現明確 fail；仍建議抽樣複核原文 syslog。")
    parts.extend(risks)

    parts.extend(["", "## 三、具體修補建議", ""])
    parts.extend([
        "- 優先處理 status=fail 控制項：先保全樣本 syslog，再開事件單追查。",
        "- 存取面：停用 Telnet、強化 AAA／個人帳，對失敗登入來源做白名單。",
        "- 惡意／埠安全：隔離命中埠、更新簽章與 port-security／802.1X。",
        "- 傳輸面：改 SNMPv3、限制 NMS 來源，複核憑證與 community。",
        "- 介面抖動：對 UPDOWN 高發埠做實體連線與變更單對帳。",
        "- 若要逐步說明修補，可再問「幫我說明修補步驟」。",
        "",
        "【報告結束】",
    ])
    return "\n".join(parts)


def sanitize_agent_chat_reply(
    reply: str,
    user_message: str,
    ot_context: str = "",
    rag_context: str = "",
) -> str:
    """對話回覆防幻覺／異常：改 grounded／修補步驟／現況報告。"""
    r = (reply or "").strip()
    casual = is_casual_chat(user_message) or is_off_topic_chat(user_message)

    # 閒聊：不再強制換成固定短答（與專業問答一樣走 LLM 結果）
    if casual and ENABLE_CASUAL_FIXED_REPLY:
        if (
            not r
            or "剛才輸出異常" in r
            or "回答格式不穩定" in r
            or _looks_like_report(r)
            or _looks_like_train_leak(r)
            or _cjk_count(r) < 12
            or re.search(r"不要補劇本|總字數\s*≤|防幻覺鐵律|智慧合規診斷", r)
            or _report_hallucinates_against_log(r, user_message)
        ):
            return build_casual_chat_reply(user_message)
        return re.sub(r"【(?:回答|報告)結束】", "", r).strip() or build_casual_chat_reply(
            user_message
        )
    if casual:
        # 只去掉結束標記；亂碼交由下方通用處理／run_llm 重寫
        r = re.sub(r"【(?:回答|報告)結束】", "", r).strip()
        if _looks_like_ollama_meta_leak(r):
            return build_casual_chat_reply(user_message)
        reply = r or reply

    # 圖表請求：勿改寫成長修補文／三卡（圖由 ensure_visual_reply 補）
    if wants_visual(user_message):
        if (
            not r
            or "剛才輸出異常" in r
            or "回答格式不穩定" in r
            or re.search(r"不要補劇本|總字數\s*≤|防幻覺鐵律", r)
        ):
            return build_ot_visual_brief_reply(user_message)
        return _polish_chat_output(reply)

    # 後處理把正文洗掉／提示詞洩漏／過短亂碼時，依問題類型補有用答案
    prompt_leaked = bool(
        re.search(
            r"不要補劇本|總字數\s*≤|防幻覺鐵律|不要使用英文|僅回答|"
            r"語言硬性|最高優先|禁止捏造|寫作指示|智慧合規診斷報告\s*$|"
            r"分析使用者|使用者本則問題|從斷點|若有對話紀錄",
            r,
        )
    ) or _looks_like_train_leak(r) or _looks_like_ollama_meta_leak(r)
    garbage_short = (
        len(r.strip()) <= 4
        or _cjk_count(r) < 18
        or bool(re.fullmatch(r"[上下左右是否好的嗯啊]+", r.strip()))
        or _looks_like_train_leak(r)
    )
    fake_metrics = _looks_like_fake_metrics_reply(r)
    # 現況類交由 LLM；僅在明顯捏造假控制項／連續假數字時才改回真實計數
    if fake_metrics and not wants_ot_status_summary(user_message):
        print("⚠️ 偵測到虛構控制項／假計數，改回真實監控筆數")
        counts = build_ot_data_counts_reply()
        if counts:
            return counts
    if (
        not r
        or "剛才輸出異常" in r
        or "回答格式不穩定" in r
        or prompt_leaked
        or garbage_short
        or (_cjk_count(r) < 20 and wants_remediation_steps(user_message))
        or (_cjk_count(r) < 40 and wants_hardening_howto(user_message))
        or (wants_data_counts(user_message) and not re.search(r"\d{2,}", r))
    ):
        if wants_hardening_howto(user_message):
            return build_hardening_howto_reply(user_message)
        if wants_data_counts(user_message):
            counts = build_ot_data_counts_reply()
            if counts:
                return counts
        if wants_remediation_steps(user_message):
            remed = build_ot_remediation_steps_reply(user_message)
            if remed:
                return remed
        # 亂碼／訓練殘留：勿原樣回傳（Qwen 常見）
        if prompt_leaked or garbage_short or not r:
            # 寒暄／離題：直接給正常打招呼，勿丟「請再問一次」
            if casual:
                print("⚠️ 閒聊輸出不穩，改用固定寒暄短答")
                return build_casual_chat_reply(user_message)
            print("⚠️ 對話輸出為亂碼／訓練殘留，改固定短答")
            return (
                "剛才這則回覆異常（疑似模型輸出不穩），已略過。"
                "請再問一次，或改問：合規現況、修補步驟，或直接貼 Cisco syslog。"
            )
        # --- 暫時註解：勿再回灌三卡現況報告 ---
        # if ENABLE_THREE_CARD_REPORT and (
        #     wants_report_format(user_message) or needs_ot_context(user_message)
        # ):
        #     situ = build_ot_situation_grounded_reply()
        #     if situ:
        #         return situ
        # --- /暫時註解 ---

    evidence = _chat_evidence_blob(user_message, ot_context, rag_context)
    if (
        not _report_hallucinates_against_log(r, evidence)
        and not prompt_leaked
        and not _looks_like_train_leak(r)
    ):
        # 現況類若仍捏造假 metrics：用真實計數提示模型失敗，改回 grounded
        if fake_metrics and wants_ot_status_summary(user_message):
            print("⚠️ 現況回答仍含虛構計數，改回真實監控筆數")
            counts = build_ot_data_counts_reply()
            if counts:
                return counts
        return _polish_chat_output(reply)

    print("⚠️ 對話回覆偵測到幻覺特徵，改寫為安全回答")
    if wants_hardening_howto(user_message):
        return build_hardening_howto_reply(user_message)
    if wants_data_counts(user_message) or fake_metrics:
        counts = build_ot_data_counts_reply()
        if counts:
            return counts
    # --- 暫時註解：對話幻覺時改灌 Cisco grounded 報告 ---
    # if ENABLE_CISCO_GROUNDED_REPORT:
    #     grounded = build_cisco_log_grounded_report(
    #         user_message,
    #         control_title="ISO 27001 日誌合規診斷",
    #     )
    #     if grounded:
    #         return grounded
    # --- /暫時註解 ---
    if wants_remediation_steps(user_message):
        remed = build_ot_remediation_steps_reply(user_message)
        if remed:
            return remed
    # --- 暫時註解：勿再回灌三卡現況報告 ---
    # if ENABLE_THREE_CARD_REPORT and (
    #     ot_context or needs_ot_context(user_message) or wants_report_format(user_message)
    # ):
    #     situ = build_ot_situation_grounded_reply()
    #     if situ:
    #         return situ
    # --- /暫時註解 ---
    return (
        "我無法根據目前提供的資料做出含具體設備／時間戳的斷言"
        "（先前草稿疑似出現訓練殘留的虛構內容，已攔截）。\n\n"
        "請貼上原始 Cisco syslog（含 `%FACILITY-SEV-MNEMONIC`），"
        "或問「目前合規現況／修補步驟」讓我依監控計數說明；"
        "我不會編造 HOSTNAME、MFG01 或假 log。"
    )


def build_metric_only_report(control_title=None, metric_summary=None, control_key=None) -> str:
    """無可用 syslog 原文時，只依量化指標寫較完整三卡（絕不虛構 log）。"""
    title = control_title or CONTROL_TITLES.get(control_key or "", "ISO 27001 控制項")
    metric = (metric_summary or "").strip()
    count = 0
    status = "n/a"
    text = ""
    m = re.search(r"count\s*=\s*(\d+)", metric, re.I)
    if m:
        count = int(m.group(1))
    sm = re.search(r"status\s*=\s*([a-z]+)", metric, re.I)
    if sm:
        status = sm.group(1).lower()
    tm = re.search(r"text\s*=\s*(.+)$", metric, re.I)
    if tm:
        text = tm.group(1).strip()

    s1 = [
        f"控制項「{title}」目前沒有可引用的 Cisco syslog 原文樣本（或樣本未含 %FACILITY 格式）。",
        f"量化摘要：count={count}，status={status}"
        + (f"，text={text[:80]}" if text else "")
        + "。",
        "以下只依計數與狀態推估風險層級，未虛構 HOSTNAME／時間戳／假 syslog。",
    ]

    if count <= 0:
        s2 = [
            "計數為 0：掃描範圍內此控制項尚無命中事件。",
            "可能代表環境乾淨，也可能是分類規則未涵蓋該訊息類型，需抽樣人工覆核。",
            "不宜解讀為「已通過完整稽核」，僅代表目前無此類事件證據。",
        ]
        s3 = [
            "維持 ot/*.txt 收集與定期掃描。",
            "抽樣人工檢視原始日誌，確認沒有漏分類的關鍵事件。",
            "有新事件時再執行個案診斷並更新基線。",
        ]
    elif control_key == "malware_defense":
        s2 = [
            f"惡意／入侵／埠安全相關事件計數約 {count} 筆（status={status}），數量偏高。",
            "雖缺原文，高計數仍應視為需追查訊號，不能寫成「無風險／無不合規」。",
            "常見來源：%FW MALWARE、%IPS、PORT_SECURITY／PSECURE、未授權 USB 等。",
            "半導體廠情境下可能影響機台網段或成為橫向移動跳板。",
        ]
        s3 = [
            "立即在原始 flash log 搜尋 MALWARE／IPS／PSECURE 並保全命中行。",
            "隔離可疑埠／主機，更新簽章與 MAC／802.1X 綁定。",
            "開立資安事件單，完成後做教訓學習與基線更新。",
            "調整樣本策略，確保 A.8.7 事件一定進入診斷 bundle。",
        ]
    elif control_key == "access_control":
        s1 = [
            f"存取／驗證相關事件約 {count} 筆（status={status}）。",
            "樣本多為 SEC_LOGIN／LOGIN_SUCCESS 等管理面登入紀錄，代表有遠端維運活動需稽核。",
            "status=fail 表示事件量偏高需複核，不是把成功登入解讀成攻擊失敗。",
        ]
        s2 = [
            f"存取／驗證事件約 {count} 筆（status={status}）。",
            "重點風險通常是共用帳、Telnet 明文、非白名單來源與失敗登入暴增。",
            "應對應 A.5.15，並與變更單／維運時段對帳。",
        ]
        s3 = [
            "禁用 Telnet、僅 SSHv2；管理 ACL 限制來源。",
            "導入 AAA 個人帳；監控非白名單 LOGIN_SUCCESS／LOGIN_FAILED。",
            "補齊 SEC_LOGIN／RADIUS／TACACS 原文樣本後做個案深挖。",
        ]
    elif control_key == "sec_gem_log":
        s2 = [
            f"傳輸／SNMP／加密相關事件約 {count} 筆（status={status}）。",
            "關注 AUTHFAIL、弱 community、憑證／IKE 失敗等 A.8.24 議題。",
        ]
        s3 = [
            "強制 SNMPv3、ACL 限制 NMS、輪替 community。",
            "檢查憑證效期與加密套件；對 AUTHFAIL 來源追查。",
        ]
    elif control_key == "recipe_audit":
        s2 = [
            f"組態／介面／syslog 類事件約 {count} 筆（status={status}）。",
            "可能含 CONFIG_I 與大量 LINK/LINEPROTO UPDOWN；需區分變更與抖動。",
        ]
        s3 = [
            "對 UPDOWN 高發埠做實體排查；CONFIG 需變更單與 archive 比對。",
            "確保樣本含 %LINK／%CONFIG 原文後再產出細節報告。",
        ]
    elif control_key == "patch_management":
        s2 = [
            f"弱點／映像／安裝相關事件約 {count} 筆（status={status}）。",
            "需確認是否於維護窗口、映像是否簽章驗證、是否有非預期重載。",
        ]
        s3 = [
            "建立修補清冊與回退映像；禁止無單生產時段升級。",
            "搜尋 BOOT／INSTALL／IOSXE 原文補充證據。",
        ]
    elif control_key == "supplier_security":
        s2 = [
            f"供應商／外部連線相關事件約 {count} 筆（status={status}）。",
            "含 CDP／LLDP／PnP／遠端 logging 等，代表外部可見性需治理。",
            "計數高≠立即遭駭，但代表通道面偏大，不宜寫「無事件」。",
        ]
        s3 = [
            "審核並關閉非必要 CDP／LLDP／PnP；遠端 log 僅送信任端。",
            "供應商遠端維運走跳板與時限帳；補齊原文樣本後複核。",
        ]
    else:
        s2 = [
            f"事件計數約 {count} 筆（status={status}）。",
            "缺原文時只能做風險分級，不能做根因定論。",
        ]
        s3 = [
            "重掃 OT 日誌並確認該控制項樣本進入 bundle。",
            "取得 %FACILITY-SEV-MNEMONIC 原文後再執行完整診斷。",
        ]
    return _build_report_cards(s1, s2, s3)


def ask_llm(log_text, control_key=None, title=None, metric_summary=None, rag_context=None):
    """
    呼叫 LLM 對傳入的 Log / 控制項現況進行智能分析與合規剖析。
    流程：先產結構化三卡事實底稿 → 像 RAG 注入 prompt → LLM 自由改寫；
    嚴重幻覺／缺陷時回退底稿原文。
    """
    control_title = title or CONTROL_TITLES.get(control_key or "", "ISO 27001 控制項")

    if is_empty_log(log_text) and not metric_summary:
        return build_metric_only_report(
            control_title=control_title,
            metric_summary="count=0, status=pass",
            control_key=control_key,
        )

    # 1. 安全截斷日誌
    log_text = log_text or ""
    if len(log_text) > MAX_INPUT_CHARS:
        log_text = log_text[:MAX_INPUT_CHARS] + "\n...[Log 過長已截斷]..."

    cisco_events = _extract_cisco_events(log_text)

    force_llm = os.environ.get("LLM_LOG_FORCE", "").strip().lower() in (
        "1", "true", "yes"
    )
    use_freewrite = bool(ENABLE_REPORT_LLM_FREEWRITE) or force_llm

    if not _llm_is_ready():
        print("⚠️ 模型未載入，改用量化模板")
        return build_metric_only_report(
            control_title=control_title,
            metric_summary=metric_summary,
            control_key=control_key,
        )

    factual_ready = _build_audit_grounded_draft(
        log_text=log_text,
        control_key=control_key,
        control_title=control_title,
        metric_summary=metric_summary,
        rag_context=rag_context,
        cisco_events=cisco_events,
    )
    draft_part = _grounded_draft_prompt_block(
        factual_ready,
        purpose="結構化診斷草稿",
        max_chars=1500,
    )

    metric_part = f"\n控制項量化摘要：{metric_summary}" if metric_summary else ""
    # 自由撰寫：RAG 僅作背景，避免模型照抄成固定答案
    if rag_context and not use_freewrite:
        rag_part = (
            "\n【同類 syslog 標準分析參考｜必須依「本次日誌」改寫】\n"
            "下列是歷史同類訊息的正確分析方向（登入≠重放攻擊；只談日誌實際出現的 facility/mnemonic）。\n"
            f"{rag_context}\n"
        )
    elif rag_context and use_freewrite:
        rag_trim = rag_context[:400] + ("…" if len(rag_context) > 400 else "")
        rag_part = (
            "\n【背景知識（可參考，禁止整段照抄；必須以本次日誌／計數為準）】\n"
            f"{rag_trim}\n"
        )
    else:
        rag_part = ""

    evidence_note = (
        "【本次待分析日誌】（以此為準；禁止捏造未出現的設備／IP／時間戳／攻擊劇情）：\n"
        f"{log_text}\n"
        if cisco_events
        else (
            "【本次無 Cisco 原文樣本】請只依下方量化摘要做合規研判與建議，"
            "禁止虛構 syslog 行或攻擊過程。\n"
        )
    )

    # 2. Prompt：自由撰寫強調「依證據自行論述」
    if use_freewrite:
        user_ask = (
            f"控制項：{control_title}\n"
            f"{metric_part}\n"
            f"{draft_part}"
            f"{rag_part}"
            f"{evidence_note}"
            "請依【結構化診斷草稿】改寫為三段："
            "（1）事件經過摘要（2）不合規／風險分析（3）具體修補建議。"
            "用你自己的語句與詳略，禁止整段照抄底稿，禁止與底稿矛盾。"
            "最後一行【報告結束】。"
        )
        system_extra = (
            "底稿是事實錨點；請改寫而非複製。"
            "禁止寫「開始輸入／開始輸出／格式撰寫」。"
            "禁止 MFG01、假 HOSTNAME、假 IP。"
            f"總字數 ≤ {MAX_OUTPUT_CHARS_REPORT if SPEED_MODE != 'turbo' else MAX_OUTPUT_CHARS}；"
            "日誌只轉述重點，禁止整段複製。"
        )
    else:
        user_ask = (
            f"控制項：{control_title}\n"
            f"{metric_part}\n"
            f"{draft_part}"
            f"{rag_part}"
            f"{evidence_note}"
            "請依【結構化診斷草稿】產出三個段落，每段 2-4 句："
            "（1）事件經過摘要（2）不合規／風險分析（3）具體修補建議。"
            "用你自己的語句改寫，禁止整段照抄。"
            "最後一行【報告結束】。"
        )
        system_extra = (
            "每段必須寫「實際觀察到的風險／事件／建議」，禁止寫「開始輸入／開始輸出／格式撰寫」。"
            f"總字數 ≤ {MAX_OUTPUT_CHARS}；日誌只轉述重點，禁止整段複製；不要輸出「## AI:」。"
            "無事件時依量化摘要評估，勿虛構攻擊。寫完立刻停。"
        )

    messages = [
        {
            "role": "system",
            "content": (
                f"{ZH_TW_OUTPUT_RULE}"
                "你是 OT/ISO 27001 資安稽核專家，專精 Cisco syslog／交換器管理面日誌分析。"
                "只根據「本次日誌／量化摘要」寫診斷；禁止套用無關 ICS 劇本（如重放攻擊、閥門控制、MFG01），"
                "除非日誌本身出現對應證據。"
                f"{OUTPUT_FORMAT_RULE}"
                f"{system_extra}"
            ),
        },
        {"role": "user", "content": user_ask},
    ]
    print(
        f"🧠 合規報告 LLM（底稿→改寫）："
        f"key={control_key} cisco={len(cisco_events)} freewrite={use_freewrite} "
        f"draft={len(factual_ready or '')}字"
    )
    reply = run_llm(
        messages,
        max_new_tokens=MAX_NEW_TOKENS_AUDIT,
        allow_continue=True,
        output_mode="report" if ENABLE_THREE_CARD_REPORT else "chat",
    )

    # 自由撰寫：僅嚴重幻覺／提示詞洩漏才回退；一般「偏弱」仍保留模型原文以保留差異
    hard_bad = _report_hallucinates_against_log(reply, log_text or metric_summary or "")
    leak_or_junk = bool(
        re.search(
            r"格式撰寫|開始輸入答案|開始輸出答案|```|\[HOSTNAME\]|MFG01|"
            r"智慧合規診斷報告.{0,12}開始",
            reply or "",
            re.I,
        )
    )
    if use_freewrite:
        if (
            hard_bad
            or leak_or_junk
            or not (reply or "").strip()
            or _audit_report_has_defects(reply or "", log_text, control_key)
        ):
            print("⚠️ LLM 嚴重幻覺／洩漏／報告缺陷，改回結構化底稿")
            return factual_ready
        return reply

    if (
        hard_bad
        or _report_is_weak(reply)
        or _report_lacks_domain_content(reply)
        or _audit_report_has_defects(reply or "", log_text, control_key)
    ):
        print("⚠️ LLM 幻覺或內容過弱，改回結構化底稿")
        return factual_ready
    return reply


def is_casual_chat(user_message: str) -> bool:
    """寒暄／身份詢問等，不應觸發 RAG 或診斷報告。"""
    t = (user_message or "").strip()
    if not t:
        return False
    compact = re.sub(r"[\s!！?？.。,~～…「」『』、]+", "", t).lower()
    casual = {
        "你好", "您好", "你好嗎", "您好嗎", "哈囉", "嗨", "在嗎", "早安", "午安", "晚安",
        "謝謝", "謝謝你", "謝謝您", "感謝", "再見", "拜拜", "掰掰", "ok", "好的", "嗯",
        "hello", "hi", "hey", "thanks", "thankyou", "bye", "howareyou",
        "你是誰", "你叫什麼", "你叫什麼名字", "你可以做什麼", "會做什麼", "你能做什麼",
        "介紹一下自己", "自我介紹", "最近好嗎", "今天好嗎", "忙嗎", "在不在",
    }
    if compact in casual:
        return True
    if len(t) <= 20 and re.fullmatch(
        r"(你好|您好|哈囉|嗨|hello|hi|hey)(嗎|呀|啊|嘛|呀啊)?[!！?？.。]*",
        t,
        re.I,
    ):
        return True
    if len(t) <= 24 and re.search(
        r"^(你是誰|你叫什麼|你可以做什麼|你能做什麼|介紹一下你自己)",
        t,
    ):
        return True
    # 「你好啊…今天幾號」勿丟進小模型（Qwen 易吐訓練殘留）
    if len(t) <= 48 and re.match(r"^(你好|您好|哈囉|嗨)", t):
        if not re.search(
            r"ISO|合規|稽核|日誌|syslog|修補|診斷|報告|控制項|cisco",
            t,
            re.I,
        ):
            return True
    if re.search(
        r"今天是?幾號|今天幾號|現在幾點|星期幾|幾月幾號|什麼日子|今天日期",
        t,
    ):
        return True
    return False


def is_off_topic_chat(user_message: str) -> bool:
    """明顯與 OT／ISO 合規無關的閒聊（天氣、心情等）→ 不走診斷三卡。"""
    t = (user_message or "").strip()
    if not t:
        return False
    # 已是合規／資安問題則不算離題（紅隊、ISO、日誌等）
    if re.search(
        r"ISO|合規|稽核|日誌|log|syslog|radius|snmp|PLC|SCADA|OT|資安|"
        r"控制項|弱點|修補|防火牆|存取|驗證|診斷|報告|監控|malware|惡意|"
        r"telnet|ssh|停用|禁用|加固|hardening|vty|明文|"
        r"筆數|事件量|cisco|%?[A-Z]+-\d+-|"
        r"紅隊|藍隊|滲透|演練|penetration|red\s*team|threat",
        t,
        re.I,
    ):
        return False
    if re.search(
        r"天氣|氣溫|下雨|晴天|陰天|風和日麗|幾度|下雨嗎|"
        r"心情|開心|難過|累了|吃飯|午餐|晚餐|早餐|喝咖啡|"
        r"早安啊|晚安啊|好笑|笑話|聊天|無聊|看電影|聽音樂|打球|旅遊|"
        r"股票|運勢|星座|八卦|遊戲|動漫",
        t,
        re.I,
    ):
        return True
    # 純寒暄（不含領域詞）且很短
    if is_casual_chat(t) and len(t) <= 24:
        return False
    return False


def build_casual_chat_reply(user_message: str) -> str:
    """閒聊／寒暄用固定短答，不經微調 LLM（避免合規腔與提示詞洩漏）。"""
    t = (user_message or "").strip()
    compact = re.sub(r"[\s!！?？.。,~～…「」『』、]+", "", t).lower()

    if re.search(r"你是誰|你叫什麼|自我介紹|介紹一下自己", t, re.I) or compact in {
        "你是誰", "你叫什麼", "你叫什麼名字", "介紹一下自己", "自我介紹",
    }:
        return (
            "我是 Semi-Shield Cyber Agent，協助 ISO 27001／OT 工控資安合規諮詢。"
            "可以問我合規現況、日誌分析、修補步驟，或直接貼 Cisco syslog。"
        )
    if re.search(r"可以做什麼|能做什麼|會做什麼|你能做", t, re.I) or compact in {
        "你可以做什麼", "會做什麼", "你能做什麼",
    }:
        return (
            "我可以幫你：\n"
            "- 看目前合規現況與控制項計數\n"
            "- 分析 Cisco syslog／存取與惡意事件\n"
            "- 說明修補步驟或畫簡單圖表\n"
            "直接說需求即可；閒聊我也行，但專業度在合規與 OT 日誌。"
        )
    if re.search(r"謝謝|感謝|thanks", t, re.I):
        return "不客氣！若還要查合規現況或分析日誌，隨時說。"
    if re.search(r"再見|拜拜|掰掰|bye", t, re.I):
        return "再見，祝監控順利。"
    if re.search(
        r"今天是?幾號|今天幾號|現在幾點|星期幾|幾月幾號|什麼日子|今天日期",
        t,
    ):
        return (
            "我這邊沒有連線即時日曆／時鐘，沒辦法告訴你精確日期或時間。"
            "若要談合規現況或 OT 日誌，直接問我就好。"
        )
    if re.search(r"天氣|氣溫|下雨|晴天|陰天|幾度", t, re.I):
        return (
            "我沒有連線即時天氣，沒辦法告訴你外面幾度或會不會下雨。"
            "若你想聊 OT／ISO 27001 合規、日誌分析或修補建議，我可以幫上忙。"
        )
    if re.search(r"心情|開心|難過|無聊|累了|笑話", t, re.I):
        return (
            "收到～我比較擅長合規與日誌分析。"
            "想輕鬆一點也可以先打聲招呼；要查監控現況隨時叫我。"
        )
    if re.search(r"吃飯|午餐|晚餐|早餐|咖啡", t, re.I):
        return "哈哈，記得適時休息。需要看合規或日誌時再叫我。"
    # 預設寒暄
    return (
        "你好！我是 Semi-Shield Cyber Agent。"
        "需要合規現況、日誌診斷或修補建議，直接問我就好。"
    )


# 暫時關閉對話三卡報告（事件摘要／風險分析／修補建議）；要恢復改 True
# --- 暫時註解：三卡報告版面異常，先全面關閉（旗標已移至檔案上方）---
# ENABLE_THREE_CARD_REPORT = False
ENABLE_CASUAL_FIXED_REPLY = False


def wants_report_format(user_message: str) -> bool:
    """
    是否應輸出三卡診斷報告。
    預設一般對話；僅在明確「要診斷／報告／分析現況或日誌」時才開三卡。
    """
    # --- 暫時註解：三卡版面異常，先全面關閉 ---
    if not ENABLE_THREE_CARD_REPORT:
        return False
    # --- /暫時註解 ---
    if is_casual_chat(user_message) or is_off_topic_chat(user_message):
        return False
    if wants_visual(user_message):
        return False
    t = (user_message or "").strip()
    if not t:
        return False

    # 知識問答 → 一般對話（不要三卡）
    if re.search(
        r"^(什麼是|何謂|介紹|請介紹|說明一下|怎麼定義|如何定義|差異是什麼|有哪些)",
        t,
    ):
        return False

    # 明確要求報告／診斷／現況
    explicit = [
        r"診斷報告", r"合規報告", r"稽核報告", r"產出報告", r"產生報告",
        r"生成報告", r"生成合規報告", r"生成稽核報告", r"幫我生成報告",
        r"出一份報告", r"寫一份報告", r"三卡", r"診斷分析",
        r"幫我診斷", r"執行診斷", r"做診斷", r"全面診斷",
        r"分析報告", r"風險報告",
        r"合規現況", r"監控現況", r"現況報告", r"現況總覽",
        r"^現況$", r"^合規$", r"目前現況", r"稽核現況",
    ]
    if any(re.search(p, t, re.I) for p in explicit):
        return True

    # 情境：針對「目前監控／日誌／合規現況」做分析檢查
    situational = [
        r"(目前|現在|今日|今天|這次).{0,12}(合規|現況|風險|缺失)(?!.*天氣)",
        r"(分析|檢查|診斷|審視|檢視).{0,16}(日誌|log|監控|控制項|合規|現況|事件)",
        r"(日誌|log|監控|syslog|radius|存取控制).{0,16}(分析|診斷|檢查|風險|問題|異常)",
        r"(有沒有|是否).{0,8}(不合規|缺失|資安風險|異常登入)",
        r"不合規", r"缺失項目", r"幫我看一下.*(日誌|現況|合規)",
        r"現況",  # 快捷按鈕「合規現況」等
    ]
    return any(re.search(p, t, re.I) for p in situational)


def wants_loose_compliance_chat(user_message: str) -> bool:
    """分析合規／控制項等，但未明確要求正式報告 → 自然對話，不要三卡。"""
    if wants_report_format(user_message):
        return False
    t = (user_message or "").strip()
    if not t:
        return False
    patterns = [
        r"分析.{0,16}(合規|控制項|稽核|缺失|現況|項目)",
        r"(合規|控制項|稽核|缺失).{0,16}分析",
        r"幫我.{0,8}(看|查|分析).{0,16}(合規|控制項|項目)",
        r"合規項目",
    ]
    return any(re.search(p, t, re.I) for p in patterns)


def needs_ot_context(user_message):
    """依關鍵字判斷是否需要下探 OT 日誌資料。"""
    if is_casual_chat(user_message) or is_off_topic_chat(user_message):
        return False
    keywords = [
        "nc", "缺失", "不符", "合規", "稽核", "日誌", "log", "威脅", "入侵",
        "隨身碟", "usb", "機台", "組態", "radius", "tacacs", "snmp", "syslog",
        "存取", "驗證", "malware", "惡意", "現況", "狀態", "控制項", "iso",
        "ot", "掃描", "資料庫", "事件", "breach", "patch", "修補",
        "telnet", "ssh", "停用", "禁用", "加固", "明文", "vty",
        "筆數", "資料量", "事件量", "多少筆", "幾筆", "總行",
        "圖表", "圖形", "視覺化", "趨勢", "統計", "chart", "pie", "bar",
        "流程圖", "架構圖", "mermaid"
    ]
    text = user_message.lower()
    return any(k in text for k in keywords)


STATUS_SCORE = {"pass": 0, "review": 1, "fail": 2, "compliant": 0, "attention": 1}

CONTROL_LABEL_ALIAS = {
    "sec_gem_log": "A.8.24",
    "recipe_audit": "A.8.19",
    "access_control": "A.5.15",
    "patch_management": "A.8.8",
    "supplier_security": "A.5.19",
    "malware_defense": "A.8.7",
}


def wants_visual(user_message):
    """判斷使用者是否明確或隱含要求圖表/圖形。"""
    keywords = [
        "圖表", "圖形", "視覺化", "趨勢圖", "統計圖", "長條圖", "圓餅圖", "折線圖",
        "雷達圖", "環圈圖", "甜甜圈", "chart", "pie", "bar", "radar", "doughnut",
        "流程圖", "架構圖", "mermaid", "畫出", "畫一個", "繪製", "可視化",
        "用圖", "做成圖", "長條", "圓餅", "狀態圖", "統計", "分布", "各種圖", "多種圖"
    ]
    text = (user_message or "").lower()
    return any(k in text for k in keywords)


def reply_mentions_chart(reply):
    """模型口頭說要畫圖、或丟了狀態表格，但未必輸出可渲染區塊。"""
    if not reply:
        return False
    text = reply.lower()
    chart_words = [
        "長條圖", "圓餅圖", "折線圖", "雷達圖", "環圈圖", "圖表", "圖形",
        "bar chart", "pie chart", "line chart", "radar", "doughnut",
        "chart", "視覺化", "bar graph"
    ]
    status_keys = [
        "sec_gem_log", "recipe_audit", "access_control",
        "patch_management", "supplier_security", "malware_defense"
    ]
    has_table = "|" in reply and ("---" in reply or "-|-" in reply)
    mentions_chart = any(w in text for w in chart_words)
    has_status_table = has_table and any(k in text for k in status_keys)
    return mentions_chart or has_status_table


def reply_has_visual_block(reply):
    """檢查模型回覆是否已含可渲染的 chart / mermaid 區塊。"""
    if not reply:
        return False
    lower = reply.lower()
    if "```chart" in lower or "```mermaid" in lower or "```json-chart" in lower:
        return True
    if '"datasets"' in lower and '"labels"' in lower and any(
        t in lower for t in ('"bar"', '"line"', '"pie"', '"doughnut"', '"radar"', '"polararea"')
    ):
        return True
    if any(lower.strip().startswith(p) or f"\n{p}" in lower for p in (
        "flowchart", "graph ", "sequencediagram"
    )):
        return True
    return False


def detect_chart_types(user_message, reply=""):
    """
    依使用者問題與模型回覆文字，決定要產生哪些圖表類型。
    回傳如 ['bar', 'pie', 'line', 'radar', 'doughnut', 'mermaid']
    """
    text = f"{user_message or ''}\n{reply or ''}".lower()
    types = []

    mapping = [
        (("長條", "bar chart", "bar graph", "柱狀"), "bar"),
        (("圓餅", "pie", "餅圖"), "pie"),
        (("折線", "趨勢", "line chart", "line graph"), "line"),
        (("雷達", "radar", "蛛網"), "radar"),
        (("環圈", "甜甜圈", "doughnut", "donut"), "doughnut"),
        (("極區", "polar"), "polarArea"),
        # 勿用「步驟」觸發 mermaid（修補步驟常誤注入流程圖導致 Syntax error）
        (("流程圖", "架構圖", "mermaid", "flowchart", "畫流程", "畫架構"), "mermaid"),
    ]
    for keys, chart_type in mapping:
        if any(k in text for k in keys):
            types.append(chart_type)

    # 「各種 / 多種 / 圖形」→ 一次產出多種 Chart；流程圖需明確要求
    want_many = any(k in text for k in (
        "各種", "多種", "多個圖", "不同圖", "圖形", "視覺化", "圖表"
    ))
    if want_many and not types:
        types = ["bar", "pie", "line", "radar", "doughnut"]
    elif want_many:
        # 有指定也再補齊常見組合
        for t in ("bar", "pie", "line", "radar"):
            if t not in types:
                types.append(t)

    # 若完全沒指定但觸發了視覺化，預設給長條 + 圓餅
    if not types:
        types = ["bar", "pie"]

    # 保序去重
    seen, ordered = set(), []
    for t in types:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def _chart_fence(payload):
    return "\n\n```chart\n" + json.dumps(payload, ensure_ascii=False) + "\n```"


def collect_metrics_series():
    """取出繪圖用 labels / counts / status_scores / status_dist。"""
    data = get_ot_monitor_data()
    if not data or (isinstance(data, dict) and "error" in data):
        return None

    metrics = data.get("metrics", {})
    if not metrics:
        return None

    labels, counts, status_scores = [], [], []
    status_dist = {"pass": 0, "review": 0, "fail": 0}
    for key, meta in metrics.items():
        labels.append(CONTROL_LABEL_ALIAS.get(key, key))
        counts.append(int(meta.get("count") or 0))
        st = str(meta.get("status", "pass")).lower()
        status_scores.append(STATUS_SCORE.get(st, 0))
        if st in status_dist:
            status_dist[st] += 1
        else:
            status_dist["pass"] += 1

    return {
        "labels": labels,
        "counts": counts,
        "status_scores": status_scores,
        "status_dist": status_dist,
    }


def build_fallback_chart_blocks(preferred_types=None):
    """依指定類型產生多種可渲染圖表（bar/pie/line/radar/doughnut/polarArea）。"""
    series = collect_metrics_series()
    if not series:
        return []

    labels = series["labels"]
    counts = series["counts"]
    status_scores = series["status_scores"]
    status_dist = series["status_dist"]
    preferred_types = preferred_types or ["bar", "pie"]

    builders = {
        "bar": lambda: {
            "type": "bar",
            "title": "OT 控制項事件量（長條圖）",
            "labels": labels,
            "datasets": [{"label": "事件數", "data": counts}],
        },
        "pie": lambda: {
            "type": "pie",
            "title": "OT 事件量占比（圓餅圖）",
            "labels": labels,
            "datasets": [{"label": "事件數", "data": [c if c > 0 else 0 for c in counts]}],
        },
        "doughnut": lambda: {
            "type": "doughnut",
            "title": "合規狀態分布（環圈圖）",
            "labels": ["pass", "review", "fail"],
            "datasets": [{"label": "控制項數", "data": [
                status_dist["pass"], status_dist["review"], status_dist["fail"]
            ]}],
        },
        "line": lambda: {
            "type": "line",
            "title": "OT 控制項事件量（折線圖）",
            "labels": labels,
            "datasets": [{"label": "事件數", "data": counts}],
        },
        "radar": lambda: {
            "type": "radar",
            "title": "合規風險雷達圖（0=pass / 1=review / 2=fail）",
            "labels": labels,
            "datasets": [{"label": "風險等級", "data": status_scores}],
        },
        "polarArea": lambda: {
            "type": "polarArea",
            "title": "OT 事件量極區圖",
            "labels": labels,
            "datasets": [{"label": "事件數", "data": counts}],
        },
    }

    blocks = []
    for t in preferred_types:
        if t == "mermaid":
            continue
        builder = builders.get(t)
        if builder:
            blocks.append(_chart_fence(builder()))
    return blocks


def build_fallback_mermaid_block(force=False, user_message=""):
    """流程/架構圖後備（標籤用雙引號、避免 / 等易炸 Mermaid 10 的字元）。"""
    if not force:
        # 僅在明確要流程／架構圖時才注入（「步驟／各種圖」不要硬塞 mermaid）
        flow_keys = ["流程圖", "架構圖", "flowchart", "mermaid", "畫流程", "畫架構"]
        text = (user_message or "").lower()
        if not any(k in text for k in flow_keys):
            return None
    return (
        "\n\n```mermaid\n"
        "flowchart TD\n"
        'A["蒐集 OT 日誌"] --> B["對照 ISO 27001 控制項"]\n'
        'B --> C{"是否存在 NC 或風險"}\n'
        'C -->|是| D["產出修補建議"]\n'
        'C -->|否| E["維持監控"]\n'
        'D --> F["驗證與複核"]\n'
        "F --> E\n"
        "```"
    )


def ensure_visual_reply(user_message, reply):
    """
    若需要視覺化但模型未輸出可解析區塊，依需求補上多種類型圖表/流程圖。
    """
    # 即使已有部分 chart，若使用者要「各種圖」且類型不足，也可再補
    preferred = detect_chart_types(user_message, reply)
    existing = reply or ""
    existing_lower = existing.lower()

    has_any_chart = reply_has_visual_block(existing)
    should_inject = wants_visual(user_message) or reply_mentions_chart(reply)
    if not should_inject and not has_any_chart:
        return reply

    # 檢查已有哪些 type，缺的再補
    missing = []
    for t in preferred:
        if t == "mermaid":
            if "```mermaid" not in existing_lower:
                missing.append(t)
        else:
            token = f'"type":"{t.lower()}"'
            # polarArea 大小寫兼容
            alt = f'"type": "{t}"'
            if token not in existing_lower and alt.lower() not in existing_lower:
                missing.append(t)

    if not missing and has_any_chart:
        return reply

    chart_types = [t for t in missing if t != "mermaid"]
    if chart_types:
        for block in build_fallback_chart_blocks(chart_types):
            reply = (reply or "") + block
    elif not has_any_chart:
        # 完全沒有圖時，至少補長條 + 圓餅
        for block in build_fallback_chart_blocks(["bar", "pie"]):
            reply = (reply or "") + block

    if "mermaid" in missing or "mermaid" in preferred:
        mermaid_block = build_fallback_mermaid_block(
            force=True,
            user_message=user_message,
        )
        if mermaid_block and "```mermaid" not in (reply or "").lower():
            reply = (reply or "") + mermaid_block

    return reply


def build_ot_context_summary():
    """掃描 OT 目錄並整理成精簡上下文給 Agent。"""
    data = get_ot_monitor_data()
    if data is None:
        return "OT 目錄不存在，尚無可用日誌。"
    if isinstance(data, dict) and "error" in data:
        return data["error"]

    metrics = data.get("metrics", {})
    logs = data.get("parsed_logs", [])[:8]
    lines = [
        "【目前 OT / ISO 27001 監控摘要】",
        "（僅下列六個控制項為真實計數；禁止新增其他控制項名稱或假數字）",
    ]
    for key, title in CONTROL_TITLES.items():
        meta = metrics.get(key) or {}
        lines.append(
            f"- {title}｜{key}：count={int(meta.get('count') or 0)}，"
            f"{meta.get('text') or 'n/a'}，status={meta.get('status') or 'n/a'}"
        )

    if logs:
        lines.append("\n【近期稽核事件】")
        for item in logs:
            lines.append(
                f"- {item.get('time')} | {item.get('file')} | "
                f"{item.get('label')} | {item.get('statusText')} | {item.get('raw')}"
            )
    else:
        lines.append("\n目前尚無解析後的事件清單。")

    summary = "\n".join(lines)
    if len(summary) > MAX_INPUT_CHARS:
        summary = summary[:MAX_INPUT_CHARS] + "\n...[摘要已截斷]..."
    return summary


CHART_FORMAT_HINT = (
    "【圖形規則】需要圖表時，先用 2-4 句繁體中文說明重點，"
    "再輸出一個 ```chart``` JSON（含 type/labels/datasets）。"
    "禁止解釋 Chart.js / chartjs API，禁止重複同一句，禁止只輸出半句英文。"
    "最後一行寫【回答結束】。"
)


_AGENT_NO_HALLUCINATION = (
    "【防幻覺鐵律】只能使用使用者訊息、監控摘要、RAG 中實際出現的內容。"
    "禁止捏造：HOSTNAME、[IP_ADDRESS]、MFG01、P01 機臺、假時間戳、假 syslog 行、"
    "未出現的 IP／設備名、異場域（汽車廠／化工廠等除非證據有）。"
    "禁止虛構控制項名稱（如 network_monitor、intrusion_detection）與假數字"
    "（如 12345、23456、45678 UPDTS）；只能使用摘要裡出現的六個控制項與真實 count。"
    "SEC_LOGIN／LOGIN_SUCCESS 不得解讀為重放攻擊或開啟閥門；"
    "UPDOWN 不得解讀為 Modbus 寫入／配方下載。"
    "證據不足就明說「無法從現有資料確認」，不要補劇本。"
)


def _normalize_chat_history(raw, max_turns: int = 6) -> list[dict]:
    """前端多輪紀錄 → OpenAI-style messages（不含本則使用者問題）。"""
    if not isinstance(raw, list) or not raw:
        return []
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = (item.get("role") or "").strip().lower()
        content = (item.get("content") or item.get("text") or "").strip()
        if not content:
            continue
        if role in ("user", "human"):
            role = "user"
        elif role in ("assistant", "agent", "bot", "ai"):
            role = "assistant"
        else:
            continue
        if role == "assistant" and content.startswith("您好！我是您的 ISO 27001"):
            continue
        limit = 600 if role == "user" else 700
        if len(content) > limit:
            content = content[:limit] + "…"
        if out and out[-1]["role"] == role:
            out[-1]["content"] = f"{out[-1]['content']}\n\n{content}".strip()
        else:
            out.append({"role": role, "content": content})
    return out[-(max(1, int(max_turns)) * 2) :]


def _wants_iso27001_topic(user_message: str) -> bool:
    """使用者是否在問 ISO/IEC 27001（含『包含哪些項目』）。"""
    t = (user_message or "").strip()
    if not t:
        return False
    return bool(
        re.search(
            r"我(要的?是|是說|指的是|問的是)\s*(iso\s*/?\s*iec\s*)?27001|"
            r"iso\s*/?\s*iec\s*27001|iso\s*27001|iso27001|(?<!\d)27001(?!\d)",
            t,
            re.I,
        )
    )


_ALLOWED_ISO27K_NUMS = frozenset(
    {27000, 27001, 27002, 27005, 27017, 27018, 27701}
)


def _extract_iso_like_numbers(text: str) -> set[int]:
    """抽出回覆中的 ISO 風格編號（含 ISO/IES 22007、裸 25701）。"""
    found: set[int] = set()
    if not text:
        return found
    for m in re.finditer(
        r"ISO\s*/?\s*I?E?[CS]?\s*[-–—]?\s*(\d{4,5})\b|(?<!\d)(2\d{4})(?!\d)",
        text,
        re.I,
    ):
        n = int(m.group(1) or m.group(2))
        # 排除明顯西元年
        if 1990 <= n <= 2100:
            continue
        found.add(n)
    return found


def _iso27001_reply_has_fake_numbers(reply: str) -> bool:
    """27001 問答中是否出現亂編標準號（22007、25701、27101…）。"""
    nums = _extract_iso_like_numbers(reply or "")
    fake = {n for n in nums if n not in _ALLOWED_ISO27K_NUMS and n >= 2000}
    if fake:
        return True
    return bool(
        re.search(
            r"ISO\s*/?\s*I?E?[CS]?\s*(?:22007|25701|22741|26004|27101|23702|"
            r"25100|23003|25000|2900|14000|9001?)|"
            r"2900\s*系列|"
            r"IE\s*CERT|ISO\s*/?\s*IE\s*CERT",
            reply or "",
            re.I,
        )
    )


def build_iso27001_prompt_knowledge(user_message: str = "") -> str:
    """
    問 27001 時附加到 prompt 的固定正確摘要（開卷），降低小模型亂編標準號。
    與 build_iso27001_overview_reply 同源，但寫成「給模型用的事實卡」。
    """
    asks_items = bool(
        re.search(r"包含|有哪些|項目|條款|控制|內容|範圍|結構", user_message or "")
    )
    common = (
        "【ISO/IEC 27001 正確事實｜必須依此回答，禁止改寫編號】\n"
        "• 正式名稱：ISO/IEC 27001（可簡稱 27001）。\n"
        "• 用途：資訊安全管理系統（ISMS）的「要求」標準；"
        "組織依此建立、實施、維護並持續改善 ISMS。\n"
        "• 同家族可提及：27000（詞彙）、27002（控制措施指引）、"
        "27005（風險管理）；不要提其他五位數亂碼。\n"
        "• 禁止：22007、25701、22741、26004、27101、23702、25100、"
        "23003、25000，以及 ISO/IES、ISO/IED 等錯字代號。\n"
        "• 本平台：以 OT／Cisco 日誌對照 27001 控制精神做合規輔助，"
        "不是另一套標準。\n"
    )
    if asks_items:
        return (
            common
            + "• 27001「包含哪些」請答兩大塊：\n"
            "  1) 管理體系要求（條文）：範圍、領導力、規劃、支援、營運、"
            "績效評估、改善；\n"
            "  2) Annex A 控制域（組織／人員／實體／技術等）；"
            "細節實務對照 27002，不要另編一套標準號。\n"
            "請用自己的話寫繁中簡答，可引用上述事實，但不要整段照抄標題格式。"
        )
    return (
        common
        + "• 若使用者只問「什麼是 27001」：用 2–5 句說明 ISMS 要求標準即可，"
        "可一句帶過 27002 是控制指引。\n"
        "請用自己的話寫繁中簡答，緊扣 27001。"
    )


def build_iso27001_overview_reply(user_message: str = "") -> str:
    """27001 知識問答的 grounded 回答（避免小模型亂編標準號／碎片輸出）。"""
    asks_items = bool(
        re.search(r"包含|有哪些|項目|條款|控制|內容|範圍|結構", user_message or "")
    )
    if asks_items:
        return (
            "ISO/IEC **27001:2022** 是資訊安全管理系統（ISMS）的**要求**標準，"
            "規定組織如何建立、實施、維護並持續改善 ISMS。\n\n"
            "**一、管理體系要求（主文條款，第 4–10 章）**\n"
            "- **第 4 章** 組織環境：確定 ISMS 範圍與利害關係人\n"
            "- **第 5 章** 領導力：高階管理承諾、資安政策、角色職責\n"
            "- **第 6 章** 規劃：風險與機會、資安目標\n"
            "- **第 7 章** 支援：資源、能力、意識、文件化資訊\n"
            "- **第 8 章** 營運：風險處置、控制措施實施\n"
            "- **第 9 章** 績效評估：監控、內部稽核、管理審查\n"
            "- **第 10 章** 改善：不符合與矯正措施、持續改善\n\n"
            "**二、Annex A 控制措施（2022 版，93 項／4 大類）**\n"
            "- **A.5** 組織控制（政策、角色、供應鏈、事件管理等）\n"
            "- **A.6** 人員控制（聘僱、培訓、離職等）\n"
            "- **A.7** 實體控制（周界、進出、設備等）\n"
            "- **A.8** 技術控制（存取、加密、日誌、弱點、組態等）\n\n"
            "**與 27002 的關係**：27001 要求「要做什麼」；"
            "**ISO/IEC 27002** 提供各控制措施的實務指引。"
            "兩者編號同屬 2700x 家族，勿與 2900、22007 等亂編號混淆。\n\n"
            "**Semi-Shield 本平台**：將 OT／Cisco syslog 對映至 Annex A 相關控制"
            "（例如 A.5.15 存取、A.8.19 組態變更、A.8.24 傳輸加密），"
            "作為合規監控與 evidence 輔助，而非取代完整 ISMS 稽核。"
        )
    return (
        "ISO/IEC **27001** 是 ISMS（資訊安全管理系統）的國際**要求**標準，"
        "用來證明組織能系統化管理資訊安全風險並持續改善。\n\n"
        "控制措施細節多見於 **ISO/IEC 27002**（指引）；"
        "27000 提供詞彙，27005 提供風險管理方法。\n\n"
        "Semi-Shield 則以 OT 日誌對照相關 Annex A 控制做合規輔助。"
        "請勿與其他亂編的 ISO 編號（如 2900 系列、22007）混淆。"
    )


def wants_iso27001_explain(user_message: str) -> bool:
    """純 27001 知識問答 → grounded 直答，不走小模型。"""
    if not _wants_iso27001_topic(user_message):
        return False
    t = (user_message or "").strip()
    if wants_visual(t) or _query_needs_log_rag(t):
        return False
    if re.search(r"分析|診斷|現況|日誌|syslog|不合規|修補|監控", t, re.I):
        return False
    if re.search(
        r"什麼是|是什麼|是甚麼|何謂|你知道|包含|有哪些|項目|條款|"
        r"範圍|結構|介紹|請說明|請解釋",
        t,
        re.I,
    ):
        return True
    return len(t) <= 28


def _is_trusted_grounded_iso27001_reply(text: str) -> bool:
    r = (text or "").strip()
    return bool(
        re.search(r"^ISO/IEC \*\*27001", r)
        or ("管理體系要求" in r and "Annex A" in r)
    )


def _iso27001_reply_is_bad(text: str) -> bool:
    """27001 回答品質過低、幻覺或格式殘留。"""
    r = (text or "").strip()
    if not r:
        return True
    if _is_trusted_grounded_iso27001_reply(r):
        return False
    if _cjk_count(r) < 80:
        return True
    if _iso27001_reply_has_fake_numbers(r):
        return True
    if re.search(r"2900\s*系列|IE\s*CERT|ISO\s*/?\s*IES", r, re.I):
        return True
    if len(re.findall(r"【(?:回答|報告)結束】", r)) >= 1:
        return True
    if re.search(r"(?m)^\s*model\s*$", r, re.I):
        return True
    if not re.search(r"27001|ISMS|資訊安全管理", r, re.I):
        return True
    if _looks_like_train_leak(r):
        return True
    return False


def ask_agent(user_message, ot_context=None, rag_context=None, chat_history=None):
    """聊天 Agent：可選擇附帶 OT 掃描結果、RAG 與多輪對話上下文。"""
    visual = wants_visual(user_message)
    history = _normalize_chat_history(chat_history)

    # 離題／寒暄＋天氣：grounded 短答（避免 Gemma 等小模型捏造台東天氣、外電廠等）
    if should_use_grounded_casual_reply(user_message) and not visual:
        print(f"💬 Agent：grounded 短答（離題／寒暄）| {user_message[:48]}")
        return build_casual_chat_reply(user_message)

    # Ollama 寒暄：小模型易複述 system／推理過程，直接固定短答
    if (
        USE_OLLAMA
        and not visual
        and is_casual_chat(user_message)
        and not re.search(r"%[A-Z0-9_-]+-\d+-", user_message or "", re.I)
    ):
        print(f"💬 Agent：Ollama 寒暄固定短答 | {user_message[:48]}")
        return build_casual_chat_reply(user_message)

    # 閒聊／離題：僅旗標全開時固定短答（舊行為）
    if (
        ENABLE_CASUAL_FIXED_REPLY
        and (not visual)
        and (is_casual_chat(user_message) or is_off_topic_chat(user_message))
    ):
        print("💬 Agent：閒聊固定短答（略過 LLM）")
        return build_casual_chat_reply(user_message)

    # 貼上 Cisco syslog 也視為診斷（即使沒寫「請分析」）
    has_cisco_log = bool(
        re.search(r"%[A-Z0-9_-]+-\d+-[A-Z0-9_]+", user_message or "", re.I)
    )
    report_mode = (wants_report_format(user_message) or has_cisco_log) and not visual
    output_mode = "report" if report_mode else "chat"
    off_topic = is_off_topic_chat(user_message)
    casual = is_casual_chat(user_message)
    rag_context = _filter_rag_for_chat(rag_context or "", user_message)
    if off_topic or casual:
        ot_context = None
        rag_context = ""

    # 有明確 Cisco 日誌時：直接 grounded，避免微調模型輸出 MFG01 幻覺
    force_llm = os.environ.get("LLM_LOG_FORCE", "").strip().lower() in (
        "1", "true", "yes"
    )
    # --- 暫時註解：日誌 grounded 也是三卡格式，一併停用 ---
    # if ENABLE_THREE_CARD_REPORT and has_cisco_log and not visual and not force_llm:
    #     grounded = build_cisco_log_grounded_report(
    #         user_message,
    #         control_title="ISO 27001 日誌合規診斷",
    #     )
    #     if grounded:
    #         print("✅ Agent：採用日誌原文 grounded 診斷（抑制 LLM 幻覺）")
    #         return grounded
    # --- /暫時註解 ---

    # 圖表請求優先：短文 + 後端補 chart（勿被修補步驟／現況長文蓋掉）
    if visual and not force_llm:
        print("✅ Agent：圖表請求，採用監控計數短文（圖由 ensure_visual_reply 補）")
        return build_ot_visual_brief_reply(user_message)

    # 「生成報告／合規現況／診斷報告」等：直接用監控 grounded，不經 LLM
    # （否則微調模型常吐出假 Admin／2023 工單狀態）
    # --- 暫時註解：三卡報告版面異常，先停用 grounded 三卡 ---
    # wants_gen_report = bool(
    #     re.search(
    #         r"生成報告|產出報告|產生報告|合規報告|稽核報告|診斷報告|"
    #         r"寫一份報告|出一份報告|分析報告|風險報告|合規現況",
    #         user_message or "",
    #         re.I,
    #     )
    # )
    # if (
    #     ENABLE_THREE_CARD_REPORT
    #     and (report_mode or wants_gen_report)
    #     and not has_cisco_log
    #     and not visual
    #     and not force_llm
    #     and (ot_context or needs_ot_context(user_message) or wants_gen_report)
    # ):
    #     situ = build_ot_situation_grounded_reply()
    #     if situ:
    #         print("✅ Agent：採用監控現況 grounded 報告（抑制 LLM 幻覺）")
    #         return situ
    # --- /暫時註解 ---

    # 「停用 Telnet／管理面加固」：固定 Cisco 步驟，避免 LLM 單字／提示詞洩漏
    if wants_hardening_howto(user_message) and not visual and not force_llm:
        print("✅ Agent：採用 grounded 管理面加固步驟")
        return build_hardening_howto_reply(user_message)

    # 「修補步驟」：直接依監控狀態給步驟，避免聊天模式淨化後變「輸出異常」
    if wants_remediation_steps(user_message) and not visual and not force_llm:
        remed = build_ot_remediation_steps_reply(user_message)
        if remed:
            print("✅ Agent：採用監控 grounded 修補步驟")
            return remed

    # 「合規現況／監控現況」：先產結構化底稿，再像 RAG 注入 LLM 改寫
    compliance_status_draft = ""
    if wants_ot_status_summary(user_message) and not visual and not force_llm:
        compliance_status_draft = build_ot_compliance_status_grounded(user_message) or ""
        if compliance_status_draft:
            print(
                f"📋 Agent：合規現況底稿 {len(compliance_status_draft)} 字 → LLM 改寫"
            )

    # 「資料筆數／事件量」：回真實掃描計數
    if wants_data_counts(user_message) and not visual and not force_llm:
        counts = build_ot_data_counts_reply()
        if counts:
            print("✅ Agent：採用監控真實筆數回覆")
            return counts

    # 知識題（27001／紅隊等）一律 LLM；事實卡僅注入 prompt + RAG，不直出固定文案

    # 現況類：強制附上真實計數給 LLM，由模型組織回答（不直接回計數表）
    if (
        wants_ot_status_summary(user_message)
        and not (ot_context or "").strip()
        and not visual
    ):
        ot_context = build_ot_context_summary()
        print("📊 Agent：現況問題已注入真實監控計數 → LLM")

    if visual:
        # 畫圖請求：要求短而完整的文字，圖表由後端 ensure_visual_reply 補強
        system_content = (
            f"{ZH_TW_OUTPUT_RULE}"
            "你是 Semi-Shield Cyber Agent。使用者要看圖表。"
            # --- 暫時註解：固定三段式格式 ---
            # f"{CHAT_OUTPUT_FORMAT_RULE}"
            # "結論寫圖表重點；說明寫合規計數／風險；建議寫可再問什麼。"
            "請只用繁體中文寫 3-6 句完整說明（合規現況與重點風險），最後一行【回答結束】。"
            # --- /暫時註解 ---
            "不要輸出 Chart.js 教學、不要重複句子、不要寫半截英文或字母亂碼。"
            "可選附上一個簡短 ```chart```；若無法穩定產出也可只寫文字。"
            f"{_AGENT_NO_HALLUCINATION}"
        )
    elif report_mode and ENABLE_THREE_CARD_REPORT:
        system_content = (
            f"{ZH_TW_OUTPUT_RULE}"
            "你是 Semi-Shield Cyber Agent，專精於 OT 工控資安與 ISO 27001 合規稽核。"
            "請以繁體中文簡潔、專業地回答，並使用 Markdown 標題與條列美化排版。"
            f"{OUTPUT_FORMAT_RULE}"
            f"總字數必須 ≤ {MAX_OUTPUT_CHARS_REPORT}。若提供監控摘要或 RAG，請轉述重點；勿虛構。"
            "知識不足時請明說。"
            "【寫作】每一段都必須從頭完整敘述；禁止用「此外／另外／同時」當段落第一句。"
            "【絕對禁令】：禁止複製／貼上 Log、RAG 原文、監控流水帳；"
            "禁止輸出「## AI:」；禁止字母亂碼、大段英文牆、簡體中文。"
            "回答第一行必須是「## 地端 LLM 智慧合規診斷報告」。"
            f"{_AGENT_NO_HALLUCINATION}"
        )
    else:
        # 對話一律用短規則（Qwen／Phi 都較穩；長規則易佔滿 turbo prompt）
        loose_compliance = wants_loose_compliance_chat(user_message)
        compliance_hint = (
            "使用者是在請你分析合規現況：請用自然段落或簡短條列說明各控制項，"
            "像同事口頭說明即可，不要寫成正式報告。"
            if loose_compliance
            else ""
        )
        if USE_OLLAMA or _is_small_or_gemma_model():
            system_content = (
                "你是 Semi-Shield Cyber Agent，專精 OT 工控資安與 ISO/IEC 27001。"
                "只用繁體中文，直接回答使用者「最新一則」問題；2-8 句或簡短條列即可。"
                "禁止捏造天氣、日期、設備 syslog 或假 ISO 編號；不知道就明說。"
                "不要輸出未閉合的【符号；勿複述本指示或分析過程。"
                f"{compliance_hint}"
            )
        else:
            system_content = (
                f"{ZH_TW_CHAT_RULE}"
                "你是 Semi-Shield Cyber Agent，專精 OT 工控資安與 ISO/IEC 27001。"
                "這是一般對話／知識問答：用 2-8 句完整中文直接回答「最新一則」使用者問題；"
                "條列最多 5 點且每點要有一句解釋。"
                "若有對話紀錄，必須依上文理解指代與更正（如「我要的是 27001」= 只要 ISO/IEC 27001）。"
                "禁止答非所問：不要改答 ISO 23003、25000、900、IE CERT，"
                "也不要亂編 22007、25701、27101 這類假標準號；"
                "提到標準時只能用真實的 27001／27002（必要時 27000／27005）。"
                "若使用者指定 27001：主題必須是 27001（ISMS 要求）；"
                "可一句說明 27002 是控制指引，但不可改答成其他編號。"
                "禁止輸出舊三卡報告格式：不要「地端 LLM 智慧合規診斷報告」標題，"
                "不要用「一、二、三」或「事件經過摘要／不合規分析／修補建議」等固定章節。"
                f"{compliance_hint}"
                "勿虛構監控事件；知識不足請明說。"
                "禁止輸出訓練題『使用者：…』、英文亂碼、單獨一行的 model／assistant。"
                "不要輸出【回答結束】或【報告結束】。總字數 ≤ 800。"
                f"{_AGENT_NO_HALLUCINATION}"
            )

    parts = []
    ollama_casual = (
        USE_OLLAMA
        and is_casual_chat(user_message)
        and not visual
        and not report_mode
        and not ot_context
        and not rag_context
    )
    if ollama_casual:
        parts.append(user_message)
    elif history:
        parts.append(
            "【多輪對話】上方 messages 已含先前來回；請承接文意回答本則，"
            "不要忽略使用者的更正或簡短指代。"
        )
    # 首句寒暄：基底 Qwen 無對話錨點時易出日文／訓練殘留；給輕量風格提示（仍走 LLM）
    elif is_casual_chat(user_message) or is_off_topic_chat(user_message):
        parts.append(
            "【首句寒暄｜風格】這是對話第一則。"
            "請只用 1-3 句自然繁體中文回覆打招呼即可；"
            "禁止日文、簡體、英文亂碼、禁止輸出「使用者：」訓練格式。"
        )
    if compliance_status_draft:
        status_cap = 1600 if re.search(
            r"詳細|深入|細部|完整|分析", user_message or "", re.I
        ) else 1200
        parts.append(
            _grounded_draft_prompt_block(
                compliance_status_draft,
                purpose="合規現況結構化底稿",
                max_chars=status_cap,
            ).strip()
        )
        parts.append(
            "【本則意圖】使用者在問合規／監控現況。"
            "請依上方底稿用自然對話口吻改寫（2-8 句或簡短條列），"
            "禁止整段照抄，禁止與底稿矛盾，禁止新增底稿沒有的控制項或假數字。"
            "禁止寫「以下是根據底稿改寫的回覆」等開場白，直接說明現況即可。"
        )
    elif ot_context:
        # 壓縮監控摘要，降低模型照抄原文與 prefill 時間
        ctx = ot_context
        # 現況類保留完整六控制項計數，避免截斷後模型亂補數字
        if wants_ot_status_summary(user_message):
            limit = max(CHAT_CTX_LIMIT, 1200)
        else:
            limit = 280 if visual else CHAT_CTX_LIMIT
        if len(ctx) > limit:
            ctx = ctx[:limit] + "\n...(摘要已截斷)..."
        if wants_ot_status_summary(user_message):
            parts.append(
                "【真實監控計數｜必須依此回答】\n"
                "下列 count／status 為系統掃描結果，數字與控制項名稱不可改、不可增刪。\n"
                "請用繁體中文說明合規現況與風險重點；可引用這些數字，但禁止捏造其他控制項"
                "（如 network_monitor）或假數字（如 12345、45678）。\n"
                f"{ctx}"
            )
        else:
            parts.append(
                "【內部監控摘要｜禁止照抄原文，只能轉述重點；禁止補造未列出的事件】\n"
                f"{ctx}"
            )
    if rag_context and not visual:
        rag_trim = rag_context
        # 知識問答可稍長，助 LLM 組織回答（仍非固定文案）
        knowledge_q = (
            wants_iso27001_explain(user_message)
            or wants_security_concept_explain(user_message)
            or _wants_iso27001_topic(user_message)
            or _wants_security_concept(user_message)
        )
        rag_cap = (
            min(RAG_CONTEXT_CHARS, 420)
            if _query_needs_log_rag(user_message or "")
            else min(RAG_CONTEXT_CHARS, 560 if knowledge_q else 360)
        )
        if len(rag_trim) > rag_cap:
            rag_trim = rag_trim[:rag_cap] + "…"
        parts.append(
            "【專業知識參考（RAG 檢索）】\n"
            "以下為 Semi-Shield 知識庫中與本題相關的 OT／ISO 27001 內容。\n"
            "請以資安顧問口吻整合重點作答：條理清楚、用繁體中文；"
            "可適度引用 Annex A 控制項編號或 Cisco 事件類型，"
            "但須轉述為你的專業建議，勿逐字照抄參考原文。\n"
            "禁止輸出「參考分析／專業參考」等標題，"
            "禁止把教材寫成「本次監控事件」，禁止捏造設備名／時間戳。\n"
            "若參考與使用者問題或監控事實衝突，以問題與【正確事實】為準。\n"
            f"{rag_trim}"
        )
    # 短更正句：明確點出 ISO/IEC 27001，降低答非所問
    um = (user_message or "").strip()
    wants_27001 = _wants_iso27001_topic(um)
    if _wants_security_concept(um):
        parts.append(build_security_concept_knowledge(um))
        parts.append(
            "【本則意圖】使用者在問資安概念；請依上方【正確事實】用繁體中文說明，"
            "可一句連結 ISO 27001 控制項；禁止「完全沒漏洞就不用投資資安」這類錯誤結論。"
        )
        print("📘 已附加資安概念 grounded 摘要至 prompt")
    if wants_27001:
        # 開卷：先附加固定正確摘要，再下意圖強制（小模型較不易亂編編號）
        parts.append(build_iso27001_prompt_knowledge(um))
        print("📘 已附加 ISO 27001 固定正確摘要至 prompt")
        asks_items = bool(
            re.search(r"包含|有哪些|項目|條款|控制|內容|範圍|結構", um)
        )
        if asks_items:
            parts.append(
                "【本則意圖｜強制】使用者在問「ISO/IEC 27001 包含哪些項目／內容」。\n"
                "請依上方【正確事實】用繁體中文回答；"
                "27001＝ISMS **要求**標準；"
                "包含管理體系條款（領導／規劃／支援／營運／績效／改善）與 Annex A 控制域；"
                "細節對照 **27002**。\n"
                "【絕對禁止】編造或改寫成 22007、25701、22741、26004、27101、23702、25100、"
                "23003、25000 等任何非 2700x 家族編號；不要寫 ISO/IES、ISO/IED 這種錯字代號。"
            )
        else:
            parts.append(
                "【本則意圖｜強制】使用者指定「ISO/IEC 27001（資訊安全管理系統／ISMS 要求）」。\n"
                "請依上方【正確事實】直接說明 27001 是什麼、用途、與本平台 OT 合規的關係；"
                "可一句帶過 27002 是控制措施指引。\n"
                "【絕對禁止】改答或亂編其他 ISO 編號（含 22007、25701、27101、23003、25000）；"
                "不要反問使用者是不是要講別的標準。"
            )
    parts.append(f"使用者本則問題：{user_message}")
    if visual:
        parts.append(
            # --- 暫時註解：固定三段式 ---
            # "請嚴格依「## 回答 → ### 結論／說明／建議」輸出；圖表會由系統自動補上。"
            "請用繁體中文說明合規狀態重點；圖表會由系統自動補上。"
            # --- /暫時註解 ---
            "不要貼日誌或 RAG 原文。不要捏造設備名。"
        )
    elif report_mode and ENABLE_THREE_CARD_REPORT:
        parts.append(
            "請直接輸出繁體中文 Markdown 報告（從「## 地端 LLM 智慧合規診斷報告」開始），"
            f"總字數 ≤ {MAX_OUTPUT_CHARS_REPORT}；三段都要寫完整；"
            "不要貼日誌／RAG／監控原文；不要捏造 syslog；最後寫【報告結束】。"
        )
    else:
        loose_hint = (
            "這是合規分析請求：用自然段落說明即可，禁止「一、二、三」章節與正式報告標題；"
            if wants_loose_compliance_chat(user_message)
            else "不要輸出舊三卡；"
        )
        parts.append(
            "請以一般對話用繁體中文簡答，緊扣本則問題；"
            f"{loose_hint}"
            "不要捏造具體設備／時間戳事件；"
            "不要輸出【回答結束】、model 等格式標記。"
        )

    messages = [{"role": "system", "content": system_content}]
    # 多輪：先放先前對話，再放本則（含監控／RAG 摘要）
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": "\n\n".join(parts)})
    if history:
        print(f"💬 多輪上下文：{len(history)} 則先前訊息")
    # 畫圖／一般聊天短答；知識問答拉長 token；報告才用 audit 預算
    knowledge_q = (
        wants_iso27001_explain(user_message)
        or wants_security_concept_explain(user_message)
        or _wants_iso27001_topic(user_message)
        or _wants_security_concept(user_message)
    )
    token_budget = MAX_NEW_TOKENS_VISUAL if visual else (
        MAX_NEW_TOKENS_AUDIT if report_mode else MAX_NEW_TOKENS
    )
    if knowledge_q and output_mode == "chat" and not visual:
        token_budget = max(
            token_budget,
            360 if _is_small_or_gemma_model() else 520,
        )
        print(f"📘 知識問答 LLM token_budget={token_budget}")
    elif (
        wants_ot_status_summary(user_message)
        and output_mode == "chat"
        and compliance_status_draft
        and not visual
    ):
        token_budget = max(
            token_budget,
            320 if _is_small_or_gemma_model() else 480,
        )
        print(f"📋 合規現況 LLM token_budget={token_budget}")
    reply = run_llm(
        messages,
        max_new_tokens=token_budget,
        allow_continue=False if USE_OLLAMA else True,
        output_mode=output_mode,
    )

    # 問 27001 卻亂編標準號／答非所問 → 重答；仍失敗則 grounded
    if wants_27001 and output_mode == "chat":
        if _iso27001_reply_is_bad(reply or ""):
            if _iso27001_reply_has_fake_numbers(reply or "") or not re.search(
                r"27001|ISMS|資訊安全管理", reply or "", re.I
            ):
                print("⚠️ 偵測到 27001 問答跑題／假標準號，強制重答…")
                retry_messages = list(messages) + [
                    {
                        "role": "user",
                        "content": (
                            "上一則出現亂編的 ISO 編號或格式殘留，無效。\n"
                            + build_iso27001_prompt_knowledge(user_message)
                            + "\n請只用繁體中文依上述事實回答 ISO/IEC **27001**；"
                            "除 27001／27002（必要時 27000／27005）外禁止其他標準編號；"
                            "不要寫 ISO/IES、2900 系列、22007、25701、27101；"
                            "不要輸出 model 或【回答結束】；不要反問。"
                        ),
                    }
                ]
                retry = run_llm(
                    retry_messages,
                    max_new_tokens=min(token_budget, 280),
                    allow_continue=False,
                    output_mode="chat",
                )
                if retry and not _iso27001_reply_is_bad(retry):
                    reply = retry
                elif _chat_grounded_fallback_enabled():
                    print("⚠️ 27001 重答仍不合格，改用 grounded（CHAT_GROUNDED_FALLBACK=1）")
                    reply = build_iso27001_overview_reply(user_message)
                else:
                    print("⚠️ 27001 重答仍不合格，保留 LLM 最佳輸出並做格式修正")
                    reply = _polish_chat_output(retry or reply or "")
            elif _chat_grounded_fallback_enabled():
                print("⚠️ 27001 LLM 輸出格式／品質不足，改用 grounded")
                reply = build_iso27001_overview_reply(user_message)
            else:
                print("⚠️ 27001 LLM 輸出品質不足，觸發第二次 LLM 重答…")
                retry2 = run_llm(
                    list(messages) + [{
                        "role": "user",
                        "content": (
                            build_iso27001_prompt_knowledge(user_message)
                            + "\n請用繁體中文重新回答上一題；"
                            "只談 ISO/IEC 27001／27002；"
                            "禁止 2900 系列等假編號；不要 model、【回答結束】。"
                        ),
                    }],
                    max_new_tokens=min(token_budget, 480),
                    allow_continue=True,
                    output_mode="chat",
                )
                reply = _polish_chat_output(retry2 or reply or "")

    if _wants_security_concept(user_message) and output_mode == "chat":
        if (
            not _is_trusted_grounded_security_reply(reply or "")
            and _security_concept_reply_is_bad(reply or "")
        ):
            if _chat_grounded_fallback_enabled():
                print("⚠️ 資安概念 LLM 輸出品質不足，改用 grounded")
                reply = build_security_concept_reply(user_message)
            else:
                print("⚠️ 資安概念 LLM 輸出品質不足，觸發 LLM 重答…")
                retry_sec = run_llm(
                    list(messages) + [{
                        "role": "user",
                        "content": (
                            build_security_concept_knowledge(user_message)
                            + "\n請用繁體中文重新回答；"
                            "依上方事實用自己的話說明；"
                            "不要 model、【回答結束】、不要「零漏洞不用資安」。"
                        ),
                    }],
                    max_new_tokens=min(token_budget, 420),
                    allow_continue=True,
                    output_mode="chat",
                )
                reply = _polish_chat_output(retry_sec or reply or "")

    if (
        wants_ot_status_summary(user_message)
        and output_mode == "chat"
        and compliance_status_draft
    ):
        status_bad = (
            not (reply or "").strip()
            or _looks_like_fake_metrics_reply(reply or "")
            or _audit_report_has_defects(reply or "", compliance_status_draft, None)
            or _cjk_count(reply or "") < 24
        )
        if status_bad:
            fb = build_ot_compliance_status_reply(user_message)
            if fb:
                print("⚠️ 合規現況 LLM 改寫不合格，改回格式化底稿")
                reply = fb

    return sanitize_agent_chat_reply(
        reply,
        user_message,
        ot_context=ot_context or "",
        rag_context=rag_context or "",
    )


def needs_rag(user_message: str) -> bool:
    """
    是否值得跑 RAG（預設否）。
    僅知識／日誌解讀／合規分析等「需要知識庫」的問題才開；
    寒暄、筆數、畫圖、純加固步驟、純現況計數、純 27001 概論不開。
    設 OT_RAG_ALWAYS=1 可恢復每題都檢索。
    """
    if _env_flag("OT_RAG_ALWAYS", default=False):
        return True
    t = (user_message or "").strip()
    if not t:
        return False
    if is_casual_chat(t) or is_off_topic_chat(t):
        return False
    # 已有 grounded／監控數字即可回答 → 不灌 RAG
    if wants_hardening_howto(t):
        return False
    if wants_data_counts(t) and not re.search(
        r"分析|風險|不合規|為什麼|含義|代表", t, re.I
    ):
        return False
    if wants_visual(t) and not re.search(
        r"分析|風險|不合規|ISO|控制項|syslog|日誌", t, re.I
    ):
        return False
    if wants_ot_status_summary(t) and not re.search(
        r"分析|為什麼|風險|缺失|不合規|知識|標準|控制項.*說明", t, re.I
    ):
        return False

    # 純 ISO 27001 知識題：RAG 輔助 LLM（非固定直答）
    if _wants_iso27001_topic(t) and not _query_needs_log_rag(t):
        return True
    # 資安概念知識題：RAG 輔助 LLM
    if _wants_security_concept(t):
        return True

    # OT／工控／Cisco 領域的風險、合規、加固問題
    if re.search(
        r"OT|工控|PLC|SCADA|ICS|DCS|Cisco|交換器|防火牆|SNMP|Telnet|SSH|RADIUS",
        t,
        re.I,
    ) and re.search(
        r"風險|為什麼|如何|怎麼|應該|建議|合規|控制|加固|修補|稽核|缺失",
        t,
        re.I,
    ):
        return True

    # 必要：貼了 Cisco syslog／明確訊息碼
    if _query_needs_log_rag(t):
        return True

    # 必要：控制項知識、日誌解讀、合規分析（不含純 27001 概論）
    patterns = [
        r"什麼是|是什麼|是甚麼|何謂|請介紹|怎麼定義|如何定義|差異是什麼|有哪些控制",
        r"A\.\d+(\.\d+)?|控制措施|Annex\s*A",
        r"控制項|合規要求|稽核重點|知識庫|歷史分析|同類事件",
        r"(分析|解讀|說明).{0,12}(日誌|log|syslog|事件|合規|風險|控制項)",
        r"(日誌|log|syslog|事件|訊息碼).{0,12}(分析|解讀|含義|代表|原因|風險)",
        r"不合規|缺失項目|風險分析|診斷報告|合規報告|幫我診斷",
        r"查一下|檢索|依知識|參考資料|標準怎麼寫",
    ]
    return any(re.search(p, t, re.I) for p in patterns)


def needs_rag_for_audit(log_text: str = "", title: str = "") -> bool:
    """監控診斷：有真實 syslog／足夠事件內容才檢索；空日誌不跑 RAG。"""
    if _env_flag("OT_RAG_ALWAYS", default=False):
        return True
    if is_empty_log(log_text):
        return False
    blob = f"{title or ''} {log_text or ''}"
    if re.search(r"%[A-Z0-9_-]+-\d+-[A-Z0-9_]+", blob, re.I):
        return True
    # 有一定資訊量才值得撈知識庫
    return (_cjk_count(blob) >= 24) or bool(re.search(r"[A-Za-z]{5,}", blob))


def retrieve_rag_for_query(query: str, *, force: bool = False):
    """統一 RAG 檢索：預設僅 needs_rag 為真時執行；force 供監控診斷使用。"""
    if not ENABLE_RAG:
        return "", [], []
    rag_service = _get_rag_service()
    if rag_service is None:
        return "", [], []
    try:
        q = (query or "").strip()
        if not q:
            return "", [], []
        if not force and not needs_rag(q):
            print(f"📚 RAG 略過（非必要問題）| {q[:60]}")
            return "", [], []

        log_mode = _query_needs_log_rag(q) or bool(force)
        # 抽出 Cisco 訊息碼；僅日誌類才加 syslog 擴寫（勿污染純知識問答）
        codes = re.findall(
            r"%?[A-Z][A-Z0-9_-]+-\d+-[A-Z][A-Z0-9_]+",
            q,
            flags=re.I,
        )
        code_hint = " ".join(codes[:6])
        if log_mode:
            q = (
                f"{q}\nCisco syslog log分析 ISO27001 {SITE_DOMAIN} "
                f"{code_hint} 事件經過 風險 修補建議"
            ).strip()
        else:
            # 知識問答：保持原問題，最多輕量加關鍵字
            q = f"{q}\nISO/IEC 27001 ISMS 控制措施 半導體 OT".strip()

        hits = rag_service.retrieve(q, top_k=3 if not log_mode else 4) or []

        log_hits, other_hits = [], []
        for h in hits:
            src = (h.get("source") or "").lower()
            blob = f"{h.get('title') or ''} {h.get('output') or ''} {h.get('snippet') or ''}"
            if _FOREIGN_SITE_RE.search(blob) and not re.search(
                r"半導體|ISO\s*27001|SEC_LOGIN|RADIUS|syslog|C9300", blob, re.I
            ):
                continue
            # 剔除 LOGIN + 重放 的有害命中
            if re.search(r"SEC_LOGIN|LOGIN_SUCCESS", blob, re.I) and re.search(
                r"重放攻擊|Replay", blob, re.I
            ):
                continue
            # 非日誌問答：丟掉三卡／劇本命中
            if not log_mode and re.search(
                r"事件經過摘要|智慧合規診斷|不合規／風險|MFG01|重放攻擊",
                blob,
                re.I,
            ):
                continue
            score = float(h.get("score") or 0)
            # 對話知識：分數太低寧可不用（亂檢索比沒有更糟）
            if not log_mode and score < 0.32:
                continue
            if src in ("ot_log", "curated") or h.get("doc_type") == "log_analysis":
                if log_mode:
                    log_hits.append(h)
                # 非日誌模式略過 log_analysis，避免劇本進 prompt
            else:
                other_hits.append(h)
        hits = (log_hits + other_hits)[: (3 if log_mode else 2)]
        if not hits:
            print("📚 RAG 無可用命中（已過濾），略過注入")
            return "", [], []

        if log_mode:
            ctx_limit = max(RAG_CONTEXT_CHARS, 720 if SPEED_MODE == "turbo" else 900)
        else:
            # 知識問答：稍長一點，助 LLM 組織專業回答
            ctx_limit = min(max(RAG_CONTEXT_CHARS, 360), 620)
        context = rag_service.format_context(hits, max_chars=ctx_limit)
        context = _sanitize_domain_text(context)
        citations = rag_service.citation_payload(hits)
        return context, citations, hits
    except Exception as e:
        print(f"⚠️ retrieve_rag_for_query 失敗：{e}")
        return "", [], []

# =======================================================
# 2. OT 目錄掃描：讀取 .txt 設備日誌（取代 manifest.json）
# =======================================================
_RE_CISCO_LOG = re.compile(
    r"^(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}(?:\s+[A-Z]+)?)\s*:\s*"
    r"%(?P<facility>[A-Z0-9_]+)-(?P<severity>\d+)-(?P<mnemonic>[A-Z0-9_]+):\s*(?P<body>.*)$"
)

# Cisco facility/關鍵字 → 控制項
_TXT_CONTROL_RULES = [
    # 存取／登入
    (
        "access_control",
        r"SEC_LOGIN|LOGIN|LOGOUT|AUTH|AAA|RADIUS|TACACS|SSH|TELNET|"
        r"LOCAL_LOGIN|LOGIN_FAILED|LOGIN_SUCCESS|TTY_EXPIRE|"
        r"DOT1X|MAB_|USER_LOCKED|AUTHEN_",
    ),
    # 傳輸／加密／SNMP
    (
        "sec_gem_log",
        r"SNMP|CRYPTO|PKI|TLS|SSL|IPSEC|IKE|SSH_SESSION|CERTIFICATE|SUDI|"
        r"COMMUNITY_MISMATCH|CERT_",
    ),
    # 弱點／韌體／映像
    (
        "patch_management",
        r"BOOT|IMAGE|VERSION|PLATFORM|SOFTWARE|UPGRADE|IOSXE|INSTALL|SMU_|FW_",
    ),
    # 供應鏈／外部連線／遠端 logging
    (
        "supplier_security",
        r"CDP|LLDP|NEIGHBOR|LOGGING\s+TO|TRAP|REMOTE.?HOST|PNP_|BGP_ADJ",
    ),
    # 惡意／異常防護
    (
        "malware_defense",
        r"IPS|IDS|MALWARE|VIRUS|THREAT|PORTSCAN|DOS|DDOS|USB_DEVICE|AV_ALERT|"
        r"HOST_ATTACK|ATTACK_DETECTED",
    ),
    # 其餘組態／介面／PoE／環境 → 組態變更稽核（A.8.19）
    (
        "recipe_audit",
        r"CONFIG|SYS-|LINK-|LINEPROTO|ILPOWER|PARSER|DUAL_ACTIVE|"
        r"SPANTREE|PORT_SECURITY|ERR_DISABLE|UPDOWN|POWER_|ENV-|FAN_|TEMP_|"
        r"RELOAD|CMD_DENIED|change-id",
    ),
]


def _parse_device_from_txt_name(filename: str):
    """
    從檔名解析設備資訊。
    例：C9300-48p_192.168.3.254_flash.txt → (C9300-48p, 192.168.3.254)
    """
    stem = Path(filename).stem
    # 去掉常見後綴
    stem = re.sub(r"_(flash|log|syslog|buffer)$", "", stem, flags=re.I)
    ip_m = re.search(r"(\d{1,3}(?:\.\d{1,3}){3})", stem)
    ip = ip_m.group(1) if ip_m else ""
    device = stem
    if ip:
        device = stem.replace(ip, "").strip("_-.")
    device = device or stem
    return device, ip


def _classify_txt_log_line(line: str):
    """依 Cisco syslog 內容對應 ISO 控制項 key；無法辨識則略過。"""
    s = (line or "").strip()
    if not s or s.startswith("Log Buffer"):
        return None

    m = _RE_CISCO_LOG.match(s)
    facility = ""
    severity = 5
    ts = ""
    body = s
    if m:
        facility = m.group("facility") or ""
        try:
            severity = int(m.group("severity"))
        except Exception:
            severity = 5
        ts = m.group("ts").strip()
        body = f"%{facility}-{severity}-{m.group('mnemonic')}: {m.group('body')}"
    else:
        # 無標準時間戳的 %FACILITY- 行
        m2 = re.search(r"%([A-Z0-9_]+)-(\d+)-([A-Z0-9_]+):", s)
        if not m2:
            return None
        facility = m2.group(1)
        try:
            severity = int(m2.group(2))
        except Exception:
            severity = 5
        body = s

    hay = f"{facility} {body}"
    key = None
    for ctrl_key, pat in _TXT_CONTROL_RULES:
        if re.search(pat, hay, re.I):
            key = ctrl_key
            break
    if not key:
        # 有 Cisco 訊息格式但未命中規則 → 歸入 syslog／組態稽核
        key = "recipe_audit"

    # 對應顯示用協議標籤
    if key == "access_control":
        proto = "RADIUS" if re.search(r"RADIUS", hay, re.I) else "SYSLOG"
        if re.search(r"TACACS", hay, re.I):
            proto = "TACACS"
    elif key == "sec_gem_log":
        proto = "SNMP"
    else:
        proto = "SYSLOG"

    mapping = PROTOCOL_CONTROL_MAP.get(proto) or PROTOCOL_CONTROL_MAP["SYSLOG"]
    # 覆寫為實際控制項（因多規則可能共用 SYSLOG 標籤）
    label_map = {
        "sec_gem_log": ("A.8.24 (Crypto)", "A.8.24 傳輸加密"),
        "recipe_audit": ("A.8.19 (Logs)", "A.8.19 組態變更"),
        "access_control": ("A.5.15 (Access)", "A.5.15 存取控制"),
        "patch_management": ("A.8.8 (Patch)", "A.8.8 弱點修補"),
        "supplier_security": ("A.5.19 (Supplier)", "A.5.19 供應鏈"),
        "malware_defense": ("A.8.7 (Malware)", "A.8.7 端點防禦"),
    }
    ctrl_id, label = label_map.get(key, (mapping["ctrl_id"], mapping["label"]))

    badge_type = "pass-bg"
    status_txt = "Compliant"
    if severity <= 2:
        badge_type = "fail-bg"
        status_txt = "Critical"
    elif severity <= 3 or key == "access_control":
        badge_type = "review-bg"
        status_txt = "Attention"
    elif re.search(r"CONFIG_I|LOGIN_FAILED|DENIED|ERR_DISABLE", hay, re.I):
        badge_type = "review-bg"
        status_txt = "Attention"

    return {
        "time": ts or "unknown",
        "facility": facility,
        "severity": severity,
        "key": key,
        "control": ctrl_id,
        "label": label,
        "type": badge_type,
        "statusText": status_txt,
        "raw": body[:300],
        "proto": proto,
    }


def _list_ot_txt_files():
    """列出 OT 目錄下所有 .txt（含子目錄）。"""
    root = Path(OT_FOLDER)
    if not root.is_dir():
        return []
    files = sorted(root.rglob("*.txt"))
    # 忽略明顯非日誌檔
    skip = {"requirements.txt", "readme.txt", "license.txt"}
    out = []
    for p in files:
        name = p.name.lower()
        if name in skip or name.startswith("expand_"):
            continue
        out.append(p)
    return out


def _parse_txt_log_file(path: Path, max_events_per_file: int = 500):
    """
    解析單一設備 .txt 日誌。
    - 全量計數（支援數萬筆）
    - 只回傳代表性樣本給前端／LLM，避免記憶體與 prompt 爆掉
    - 另在掃描時為各控制項保留少量原文樣本（避免 A.5.19 等有計數卻無樣本）
    """
    device, ip = _parse_device_from_txt_name(path.name)
    counts_by_key = {k: 0 for k in CONTROL_TITLES}
    samples_by_key = {k: [] for k in CONTROL_TITLES}
    recent = []
    priority = []
    total_lines = 0
    event_lines = 0

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                total_lines += 1
                ev = _classify_txt_log_line(line)
                if not ev:
                    continue
                event_lines += 1
                key = ev.get("key")
                if key in counts_by_key:
                    counts_by_key[key] += 1
                ev["file"] = path.name
                ev["device"] = device
                ev["ip"] = ip
                ev["raw"] = f"[{device}] {ev['raw']}"

                # 全量掃描時就收各控制項樣本（不依賴後面截斷的 samples 列表）
                if key in samples_by_key and len(samples_by_key[key]) < 8:
                    samples_by_key[key].append(ev)

                recent.append(ev)
                if len(recent) > max_events_per_file * 2:
                    recent = recent[-max_events_per_file:]

                is_prio = (
                    key in (
                        "access_control",
                        "sec_gem_log",
                        "malware_defense",
                        "supplier_security",
                    )
                    or ev.get("severity", 5) <= 3
                    or re.search(
                        r"CONFIG_I|LOGIN_FAILED|DENIED|RADIUS|SNMP-3|MALWARE|"
                        r"PSECURE|CDP|LLDP|NEIGHBOR",
                        ev.get("raw", ""),
                        re.I,
                    )
                )
                if is_prio and len(priority) < max_events_per_file:
                    priority.append(ev)
    except Exception as e:
        print(f"⚠️ 讀取 TXT 失敗 {path}: {e}")
        return [], {
            "file": path.name,
            "device": device,
            "ip": ip,
            "error": str(e),
            "counts_by_key": counts_by_key,
            "samples_by_key": samples_by_key,
        }

    # 樣本：優先事件 + 尾端近期，去重
    seen = set()
    samples = []
    for ev in priority + recent[-max_events_per_file:]:
        sig = (ev.get("time"), ev.get("raw"))
        if sig in seen:
            continue
        seen.add(sig)
        samples.append(ev)
        if len(samples) >= max_events_per_file:
            break

    meta = {
        "file": path.name,
        "device": device,
        "ip": ip,
        "total_lines": total_lines,
        "event_lines": event_lines,
        "sample_size": len(samples),
        "counts_by_key": counts_by_key,
        "samples_by_key": samples_by_key,
    }
    if event_lines > len(samples):
        meta["truncated_to"] = len(samples)
    return samples, meta


def build_control_log_bundle(
    parsed_logs_list,
    metrics,
    max_lines=8,
    samples_by_key=None,
):
    """為各控制項組裝精簡日誌摘要（優先用專桶樣本，避免被總表 180 筆擠掉）。"""
    bundles = {}
    samples_by_key = samples_by_key or {}
    for key in CONTROL_TITLES:
        matched = [x for x in parsed_logs_list if x.get("key") == key]
        if len(matched) < max_lines and samples_by_key.get(key):
            # 補齊各控制項專屬樣本（解決：計數很多但總表抽不到）
            seen = {(m.get("time"), m.get("raw")) for m in matched}
            for ev in samples_by_key[key]:
                sig = (ev.get("time"), ev.get("raw"))
                if sig in seen:
                    continue
                matched.append(ev)
                seen.add(sig)
                if len(matched) >= max_lines * 2:
                    break
        metric = metrics.get(key, {})
        full_count = int(metric.get("count", len(matched)) or 0)
        metric_summary = (
            f"count={full_count}, "
            f"status={metric.get('status', 'n/a')}, "
            f"text={metric.get('text', 'n/a')}"
        )
        if matched:
            ranked = sorted(
                matched,
                key=lambda x: (x.get("severity", 5), x.get("time", "")),
            )
            lines = [m["raw"] for m in ranked[:max_lines]]
            log_text = "\n".join(lines)
            omitted = max(0, full_count - len(lines))
            if omitted > 0:
                log_text += f"\n...另有 {omitted} 筆同類事件已省略..."
        elif full_count > 0:
            log_text = (
                f"（全量計數 {full_count} 筆，目前無可用原文樣本）\n"
                "請只依量化指標評估；禁止虛構 syslog／HOSTNAME／IP 佔位內容。"
            )
        else:
            log_text = "目前該控制項沒有相關事件行數據。"

        bundles[key] = {
            "title": CONTROL_TITLES[key],
            "log": log_text,
            "metric_summary": metric_summary,
            "event_count": full_count,
        }
    return bundles


def scan_ot_directory():
    """掃描 OT 目錄下 *.txt 設備日誌，組裝 metrics／事件／控制項摘要。"""
    if not os.path.exists(OT_FOLDER):
        return None

    txt_files = _list_ot_txt_files()
    parsed_logs_list = []
    file_metas = []
    samples_by_key = {k: [] for k in CONTROL_TITLES}

    metrics = {
        "sec_gem_log": {"count": 0, "text": "0 SECURE", "status": "pass"},
        "recipe_audit": {"count": 0, "text": "0 MINOR NC", "status": "pass"},
        "access_control": {"count": 0, "text": "0 BREACH", "status": "pass"},
        "patch_management": {"count": 0, "text": "0 UNPATCHED", "status": "pass"},
        "supplier_security": {"count": 0, "text": "0 WARNINGS", "status": "pass"},
        "malware_defense": {"count": 0, "text": "0 INFECTIONS", "status": "pass"},
    }

    if not txt_files:
        return {
            "error": (
                f"在 {OT_FOLDER} 目錄下找不到 .txt 日誌檔。"
                "請放置如 C9300-xxx_192.168.x.x_flash.txt 的設備日誌。"
            )
        }

    try:
        for path in txt_files:
            events, meta, from_cache = _parse_txt_log_file_cached(path)
            file_metas.append(meta)
            parsed_logs_list.extend(events)
            # 全量計數（非僅樣本）
            for k, n in (meta.get("counts_by_key") or {}).items():
                if k in metrics:
                    metrics[k]["count"] += int(n or 0)
            # 各控制項專桶樣本：優先用全量掃描時留下的桶（含 CDP／供應商等）
            for k, evs in (meta.get("samples_by_key") or {}).items():
                if k not in samples_by_key:
                    continue
                for ev in evs:
                    if len(samples_by_key[k]) >= 16:
                        break
                    samples_by_key[k].append(ev)
            # 後備：截斷後的 events 列表再補一輪
            for ev in events:
                k = ev.get("key")
                if k in samples_by_key and len(samples_by_key[k]) < 16:
                    samples_by_key[k].append(ev)
            if not from_cache:
                print(
                    f"📄 TXT 來源：{path.name} → 事件 {meta.get('event_lines', 0)} 筆"
                    f"（樣本 {meta.get('sample_size', len(events))}，設備 {meta.get('device')}）"
                )
    except Exception as e:
        print(f"解析 TXT 日誌失敗: {e}")
        return {"error": f"解析 TXT 日誌失敗：{str(e)}"}

    # 狀態文案（門檻依 TXT 事件量調整；5 萬筆規模）
    metrics["sec_gem_log"]["text"] = f"SECURE ({metrics['sec_gem_log']['count']})"
    metrics["sec_gem_log"]["status"] = (
        "review" if metrics["sec_gem_log"]["count"] > 2000 else "pass"
    )

    metrics["recipe_audit"]["text"] = f"{metrics['recipe_audit']['count']} SYSLOGS"
    metrics["recipe_audit"]["status"] = (
        "review" if metrics["recipe_audit"]["count"] > 20000 else "pass"
    )

    metrics["access_control"]["text"] = f"{metrics['access_control']['count']} AUTHS"
    ac = metrics["access_control"]["count"]
    metrics["access_control"]["status"] = (
        "fail" if ac > 8000 else ("review" if ac > 2000 else "pass")
    )

    metrics["patch_management"]["text"] = (
        f"PATCH HINTS ({metrics['patch_management']['count']})"
    )
    metrics["patch_management"]["status"] = (
        "review" if metrics["patch_management"]["count"] > 100 else "pass"
    )

    metrics["supplier_security"]["text"] = (
        f"EXTERNAL ({metrics['supplier_security']['count']})"
    )
    metrics["supplier_security"]["status"] = (
        "review" if metrics["supplier_security"]["count"] > 1500 else "pass"
    )

    metrics["malware_defense"]["text"] = (
        f"{metrics['malware_defense']['count']} INFECTION"
    )
    metrics["malware_defense"]["status"] = (
        "fail" if metrics["malware_defense"]["count"] > 0 else "pass"
    )

    # 前端事件列表：保留較新／較重要，上限避免塞爆
    by_time = sorted(parsed_logs_list, key=lambda x: x.get("time", ""), reverse=True)
    priority, rest = [], []
    for e in by_time:
        if (
            e.get("key") in ("access_control", "sec_gem_log")
            or e.get("severity", 5) <= 3
            or re.search(r"CONFIG_I|LOGIN|DENIED", e.get("raw", ""), re.I)
        ):
            priority.append(e)
        else:
            rest.append(e)
    display_logs = (priority + rest)[:180]

    devices = sorted({m.get("device") for m in file_metas if m.get("device")})
    summary_obj = {
        "source": "txt",
        "folder": OT_FOLDER,
        "num_files": len(file_metas),
        "num_devices": len(devices),
        "device_ids": devices,
        "files": file_metas,
        "counts": {k: metrics[k]["count"] for k in metrics},
    }
    all_logs_content = json.dumps(summary_obj, indent=2, ensure_ascii=False)

    control_bundles = build_control_log_bundle(
        display_logs,
        metrics,
        samples_by_key=samples_by_key,
    )

    return {
        "metrics": metrics,
        "parsed_logs": display_logs,
        "all_logs_content": all_logs_content,
        "control_bundles": control_bundles,
        "compliance_principle": "No Evidence, No Compliance Claim",
    }


def _attach_evidence_and_reviews(monitor_data: dict) -> dict:
    """為各控制項 bundle 註冊 evidence_id；review/fail 自動進 Human Review Queue。"""
    if not isinstance(monitor_data, dict) or monitor_data.get("error"):
        return monitor_data
    bundles = monitor_data.get("control_bundles") or {}
    metrics = monitor_data.get("metrics") or {}
    evidence_map: dict[str, str] = {}
    for key, bundle in bundles.items():
        try:
            ev = register_control_bundle_evidence(key, bundle)
            eid = ev.get("evidence_id") or ""
            bundle["evidence_id"] = eid
            evidence_map[key] = eid
            st = (metrics.get(key) or {}).get("status", "pass")
            if st in ("review", "fail"):
                enqueue_review(
                    item_type="control_status",
                    title=f"{bundle.get('title', key)} — 自動化狀態 {st}",
                    summary=bundle.get("metric_summary") or "",
                    control_key=key,
                    evidence_id=eid,
                    priority="high" if st == "fail" else "normal",
                    metadata={"status": st, "event_count": bundle.get("event_count")},
                )
        except Exception as e:
            print(f"⚠️ evidence 註冊略過 {key}: {e}")
    monitor_data["evidence_map"] = evidence_map
    monitor_data["compliance_coverage"] = coverage_stats()
    return monitor_data


# OT 掃描結果快取：檔案 mtime/size 未變則直接回傳，避免輪詢反覆掃 10 萬行
_OT_SCAN_CACHE = {"sig": None, "data": None, "ts": 0.0}
_OT_FILE_PARSE_CACHE: dict = {}  # path -> {mtime, size, events, meta}
_ot_scan_lock = threading.Lock()


def _ot_files_signature() -> str:
    parts = []
    for p in _list_ot_txt_files():
        try:
            st = p.stat()
            parts.append(f"{p.name}:{int(st.st_mtime)}:{st.st_size}")
        except OSError:
            parts.append(str(p.name))
    return "|".join(parts)


def _parse_txt_log_file_cached(path: Path, max_events_per_file: int = 500):
    """單一檔案解析快取：mtime/size 未變則复用，避免雙檔每次都重讀。"""
    key = str(path.resolve()) if path.exists() else str(path)
    try:
        st = path.stat()
        mtime = int(st.st_mtime)
        size = int(st.st_size)
    except OSError:
        return _parse_txt_log_file(path, max_events_per_file=max_events_per_file) + (False,)

    hit = _OT_FILE_PARSE_CACHE.get(key)
    if hit and hit.get("mtime") == mtime and hit.get("size") == size:
        return hit["events"], hit["meta"], True

    events, meta = _parse_txt_log_file(path, max_events_per_file=max_events_per_file)
    _OT_FILE_PARSE_CACHE[key] = {
        "mtime": mtime,
        "size": size,
        "events": events,
        "meta": meta,
    }
    return events, meta, False


def get_ot_monitor_data(force: bool = False):
    """帶快取的 OT 監控資料（給 API／Agent 共用）。"""
    import time

    with _ot_scan_lock:
        now = time.time()
        sig = _ot_files_signature()
        cached = _OT_SCAN_CACHE
        ttl = float(OT_CACHE_TTL or 0)
        age = now - float(cached["ts"] or 0)
        # 檔案未變 → 直接用快取（不再每 15 秒重掃）
        # TTL>0 時可強制週期重掃；預設 0 關閉
        sig_ok = cached["data"] is not None and cached["sig"] == sig
        ttl_expired = ttl > 0 and age >= ttl
        if not force and sig_ok and not ttl_expired:
            return cached["data"]

        data = scan_ot_directory()
        if isinstance(data, dict) and "error" not in data:
            data = _attach_evidence_and_reviews(data)
            _OT_SCAN_CACHE["sig"] = sig
            _OT_SCAN_CACHE["data"] = data
            _OT_SCAN_CACHE["ts"] = now
            print(f"📦 OT 監控快取已更新（{len(_list_ot_txt_files())} 檔）")
        return data


# =======================================================
# 3. 前端頁面與 API 路由設定
# =======================================================
@app.route('/')
@app.route('/platform')
def serve_platform_page():
    """整合平台首頁：監控戰情 + AI 對話。"""
    return send_from_directory(WEB_DIR, 'platform.html')


@app.route('/chat')
def serve_chat_page():
    """提供前端聊天介面網頁（可被整合平台嵌入）。"""
    return send_from_directory(WEB_DIR, 'agent_chat.html')


@app.route('/monitor')
def serve_monitor_page():
    """提供原有稽核監控儀表板（可被整合平台嵌入）。"""
    return send_from_directory(WEB_DIR, 'OT.html')


@app.route('/api/monitor/data', methods=['GET'])
def get_monitor_data():
    force = str(request.args.get("force") or "").strip().lower() in ("1", "true", "yes")
    data = get_ot_monitor_data(force=force)
    if isinstance(data, dict) and "error" in data:
        return jsonify(data), 404
    return jsonify(data)


@app.route('/api/audit/analyze', methods=['POST'])
def analyze_log_with_llm():
    """
    單控制項地端診斷。
    Body 可為：
      { "log": "..." }
      或 { "control_key": "access_control", "log": "...", "title": "...", "metric_summary": "..." }
    若只給 control_key，後端會自動從 OT 掃描組裝摘要。
    """
    req_data = request.get_json(silent=True) or {}
    control_key = req_data.get("control_key")
    log_content = req_data.get("log")
    title = req_data.get("title")
    metric_summary = req_data.get("metric_summary")

    want_model = (req_data.get("model") or req_data.get("llm_model") or "").strip()
    if want_model:
        sw = switch_llm_model(want_model)
        if not sw.get("ok"):
            return jsonify({
                "error": sw.get("error") or "模型切換失敗",
                "status": "error",
                "llm_model": _current_llm_info(),
            }), 503

    # 若前端只丟 control_key，或 log 為空，自動補齊最新 bundle
    if control_key or is_empty_log(log_content):
        data = get_ot_monitor_data()
        if isinstance(data, dict) and "error" not in data:
            bundles = data.get("control_bundles") or {}
            key = control_key or req_data.get("key")
            if key and key in bundles:
                bundle = bundles[key]
                log_content = log_content if not is_empty_log(log_content) else bundle["log"]
                title = title or bundle["title"]
                metric_summary = metric_summary or bundle["metric_summary"]
                control_key = key

    if log_content is None and not control_key:
        return jsonify({
            "error": "請傳入 {'log': '...'} 或 {'control_key': 'access_control'}"
        }), 400

    if not _llm_is_ready():
        err = "Ollama 未連線" if USE_OLLAMA else "模型未成功載入"
        return jsonify({
            "error": f"{err}，無法提供 AI 診斷。請檢查後端啟動日誌。",
            "status": "error"
        }), 503

    # RAG：僅真實 syslog／有內容的診斷才檢索
    rag_context, rag_citations = "", []
    if needs_rag_for_audit(str(log_content or ""), title or control_key or ""):
        rag_query = f"{title or control_key or ''} ISO 27001 {str(log_content or '')[:240]}"
        rag_context, rag_citations, _ = retrieve_rag_for_query(rag_query, force=True)
    else:
        print(f"📚 RAG 略過（單項診斷無必要）| key={control_key}")

    print(f"🔍 單項診斷: key={control_key} title={title} rag_hits={len(rag_citations)}")
    ai_analysis = ask_llm(
        log_content,
        control_key=control_key,
        title=title,
        metric_summary=metric_summary,
        rag_context=rag_context or None,
    )

    evidence_id = ""
    if control_key:
        try:
            bundle = {
                "title": title or CONTROL_TITLES.get(control_key, control_key),
                "log": str(log_content or "")[:500],
                "metric_summary": metric_summary or "",
                "event_count": 0,
            }
            ev = register_control_bundle_evidence(control_key, bundle)
            evidence_id = ev.get("evidence_id") or ""
            enqueue_review(
                item_type="llm_diagnosis",
                title=f"LLM 診斷待覆核：{control_key}",
                summary=(ai_analysis or "")[:800],
                control_key=control_key or "",
                evidence_id=evidence_id,
                source="audit_analyze",
                priority="normal",
                metadata={"rag_hits": len(rag_citations)},
            )
        except Exception as e:
            print(f"⚠️ audit evidence/review: {e}")

    return jsonify({
        "status": "success",
        "control_key": control_key,
        "raw_log": log_content,
        "ai_analysis": ai_analysis,
        "evidence_id": evidence_id,
        "compliance_principle": "No Evidence, No Compliance Claim",
        "guardrail": {"skipped": True, "reason": "監控戰情略過輸入護欄；輸出建議 Human Review"},
        "human_review": {"recommended": True, "reason": "LLM 診斷需分析師覆核後方可作為稽核證據"},
        "rag_enabled": _rag_feature_enabled(),
        "rag_citations": rag_citations,
        "llm_model": _current_llm_info(),
    })


@app.route('/api/audit/analyze_all', methods=['POST'])
def analyze_all_logs():
    """
    批次診斷。Body 可為：
      { "items": { "access_control": "log text", ... } }
      或 { "keys": ["access_control", ...] } / 空 body → 自動掃描全部控制項
    """
    req_data = request.get_json(silent=True) or {}
    items = req_data.get("items")
    keys = req_data.get("keys")

    want_model = (req_data.get("model") or req_data.get("llm_model") or "").strip()
    if want_model:
        sw = switch_llm_model(want_model)
        if not sw.get("ok"):
            return jsonify({
                "error": sw.get("error") or "模型切換失敗",
                "status": "error",
                "llm_model": _current_llm_info(),
            }), 503

    data = get_ot_monitor_data()
    if data is None:
        return jsonify({"error": "OT 目錄不存在"}), 404
    if isinstance(data, dict) and "error" in data:
        return jsonify(data), 404

    bundles = data.get("control_bundles") or {}

    # 正規化待分析清單：{ key: {log, title, metric_summary} }
    jobs = {}
    if isinstance(items, dict) and items:
        for key, value in items.items():
            if isinstance(value, dict):
                jobs[key] = {
                    "log": value.get("log"),
                    "title": value.get("title") or CONTROL_TITLES.get(key),
                    "metric_summary": value.get("metric_summary"),
                }
            else:
                bundle = bundles.get(key, {})
                jobs[key] = {
                    "log": value if not is_empty_log(value) else bundle.get("log"),
                    "title": bundle.get("title") or CONTROL_TITLES.get(key),
                    "metric_summary": bundle.get("metric_summary"),
                }
    else:
        target_keys = keys if isinstance(keys, list) and keys else list(CONTROL_TITLES.keys())
        for key in target_keys:
            bundle = bundles.get(key, {})
            jobs[key] = {
                "log": bundle.get("log", "目前該控制項沒有相關事件行數據。"),
                "title": bundle.get("title") or CONTROL_TITLES.get(key),
                "metric_summary": bundle.get("metric_summary"),
            }

    if not _llm_is_ready():
        err = "Ollama 未連線" if USE_OLLAMA else "模型未成功載入"
        return jsonify({
            "error": f"{err}，無法提供 AI 診斷。請檢查後端啟動日誌。",
            "status": "error"
        }), 503

    results = {}
    rag_meta = {}
    for key, job in jobs.items():
        print(f"正在分析控制項: {key} ...")
        # RAG：有真實事件才檢索，空控制項不灌知識庫
        log_body = str(job.get("log") or "")
        rag_context, rag_citations = "", []
        if needs_rag_for_audit(log_body, job.get("title") or key):
            rag_query = f"{job.get('title') or key} ISO 27001 {log_body[:240]}"
            rag_context, rag_citations, _ = retrieve_rag_for_query(
                rag_query, force=True
            )
        else:
            print(f"📚 RAG 略過（控制項無必要）| {key}")
        analysis = ask_llm(
            job.get("log"),
            control_key=key,
            title=job.get("title"),
            metric_summary=job.get("metric_summary"),
            rag_context=rag_context or None,
        )
        results[key] = analysis
        rag_meta[key] = {
            "blocked": False,
            "output_redacted": False,
            "guardrail": {"skipped": True, "reason": "監控戰情略過護欄"},
            "rag_citations": rag_citations,
        }

    return jsonify({
        "status": "success",
        "results": results,
        "rag_enabled": _rag_feature_enabled(),
        "meta": rag_meta,
        "llm_model": _current_llm_info(),
    })


@app.route('/api/agent/chat', methods=['POST'])
def agent_chat():
    """前端聊天介面：護欄 → RAG 檢索 → LLM → 圖表補強 → 輸出脫敏。"""
    req_data = request.get_json(silent=True) or {}
    user_message = (req_data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "請傳入 JSON：{'message': '你的問題'}"}), 400

    # 可選：請求時指定模型（與功能選單一致；已是目前模型則略過）
    want_model = (req_data.get("model") or req_data.get("llm_model") or "").strip()
    if want_model:
        sw = switch_llm_model(want_model)
        if not sw.get("ok"):
            return jsonify({
                "error": sw.get("error") or "模型切換失敗",
                "status": "error",
                "llm_model": _current_llm_info(),
            }), 503

    # 1) 輸入護欄
    guard = guardrail_service.check_input(user_message)
    if guard.get("blocked"):
        print(f"🛡️ 護欄攔截: {guard.get('reason')} | {user_message[:80]}")
        try:
            enqueue_review(
                item_type="guardrail_block",
                title="護欄攔截請求",
                summary=user_message[:500],
                source="agent_chat",
                priority="high",
                metadata={"guard": guard},
            )
        except Exception:
            pass
        return jsonify({
            "status": "blocked",
            "reply": guardrail_service.block_message(guard),
            "tool_called": False,
            "tool_name": "guardrail",
            "tool_query": guard.get("mode"),
            "guardrail": guard,
            "human_review": {"recommended": True, "reason": "已進 Review Queue 供分析師覆核"},
            "rag_enabled": _rag_feature_enabled(),
            "rag_citations": [],
            "llm_model": _current_llm_info(),
        }), 200

    human_review_hint = None
    if guard.get("human_review_recommended"):
        human_review_hint = {
            "recommended": True,
            "reason": guard.get("human_review_reason") or "邊界分數，建議分析師覆核",
        }

    tool_called = False
    tool_name = None
    tool_query = None
    ot_context = None
    report_mode = wants_report_format(user_message)

    # RAG 檢索（聊天）：僅必要問題才導入
    rag_context, rag_citations = "", []
    if ENABLE_RAG and needs_rag(user_message):
        rag_context, rag_citations, _ = retrieve_rag_for_query(
            user_message, force=True
        )
        if rag_citations:
            tool_called = True
            tool_name = "rag_retrieve"
            tool_query = f"top_k={len(rag_citations)}"
            print(f"📚 RAG hits={len(rag_citations)} | {user_message[:80]}")
        else:
            print(f"📚 RAG 無命中 | {user_message[:80]}")
        if rag_context:
            rag_context = _filter_rag_for_chat(rag_context, user_message)
    elif ENABLE_RAG:
        print(f"📚 RAG 略過（非必要問題）| {user_message[:60]}")

    # 寒暄／離題：與一般問答相同，一律進 LLM（不再固定短答略過）
    if ENABLE_CASUAL_FIXED_REPLY and (
        is_casual_chat(user_message) or is_off_topic_chat(user_message)
    ):
        print(f"💬 閒聊模式（固定短答，略過 LLM／報告）| {user_message[:40]}")
        reply = build_casual_chat_reply(user_message)
        reply, redacted = guardrail_service.sanitize_output(reply)
        return jsonify({
            "status": "success",
            "reply": reply,
            "tool_called": tool_called,
            "tool_name": tool_name,
            "tool_query": tool_query,
            "guardrail": {
                "blocked": False,
                "mode": guard.get("mode") or guardrail_service.mode,
                "output_redacted": redacted,
                "safe_prob": guard.get("safe_prob", 1.0),
                "unsafe_prob": guard.get("unsafe_prob", 0.0),
                "reason": "閒聊固定短答",
            },
            "rag_enabled": _rag_feature_enabled(),
            "rag_citations": rag_citations,
            "llm_model": _current_llm_info(),
        }), 200

    if needs_ot_context(user_message) or wants_ot_status_summary(user_message):
        tool_called = True
        tool_name = f"{tool_name}+scan_ot_directory" if tool_name else "scan_ot_directory"
        tool_query = (
            f"{tool_query}; metrics + recent_parsed_logs"
            if tool_query
            else "metrics + recent_parsed_logs"
        )
        print(f"🤖 Agent Tool Call: scan_ot_directory | 問題: {user_message[:80]}")
        ot_context = build_ot_context_summary()

    # 3) LLM 生成前：若尚未探測，先隱形問「你好」並丟棄（燒掉冷啟動第一則）
    try:
        _prime_llm_chat()
    except Exception as _prime_err:
        print(f"⚠️ 請求前對話探測略過：{_prime_err}")

    # 3b) LLM 生成（帶多輪 history，避免「我是說 27001」這類接話答非所問）
    chat_history = req_data.get("history") or req_data.get("messages") or []
    reply = ask_agent(
        user_message,
        ot_context=ot_context,
        rag_context=rag_context or None,
        chat_history=chat_history,
    )
    # 雙保險：對整段證據再跑一次防幻覺改寫
    reply = sanitize_agent_chat_reply(
        reply,
        user_message,
        ot_context=ot_context or "",
        rag_context=rag_context or "",
    )
    reply = ensure_visual_reply(user_message, reply)
    # 三卡關閉時一律走 chat 淨化，再強制收斂成結論／說明／建議
    use_report_norm = bool(ENABLE_THREE_CARD_REPORT) and (
        report_mode
        or _looks_like_report(reply)
        or bool(
            re.search(r"%[A-Z0-9_-]+-\d+-[A-Z0-9_]+", user_message or "", re.I)
        )
    )
    reply = normalize_llm_output(
        reply,
        mode="report" if use_report_norm else "chat",
    )
    # normalize 後可能又露出模板殘留，再掃一次
    reply = sanitize_agent_chat_reply(
        reply,
        user_message,
        ot_context=ot_context or "",
        rag_context=rag_context or "",
    )
    reply = _finalize_chat_reply(reply, user_message)
    # --- 暫時註解：對話固定輸出（結論／說明／建議）收斂 ---
    # reply = ensure_fixed_chat_format(reply, user_message)
    # --- /暫時註解（恢復時取消註解，並將 ENABLE_FIXED_CHAT_FORMAT=True）---

    # 4) 輸出脫敏
    reply, redacted = guardrail_service.sanitize_output(reply)

    return jsonify({
        "status": "success",
        "reply": reply,
        "tool_called": tool_called,
        "tool_name": tool_name,
        "tool_query": tool_query,
        "guardrail": {
            "blocked": False,
            "mode": guard.get("mode") or guardrail_service.mode,
            "output_redacted": redacted,
            "safe_prob": guard.get("safe_prob", 1.0),
            "unsafe_prob": guard.get("unsafe_prob", 0.0),
            "reason": guard.get("reason"),
        },
        "human_review": human_review_hint,
        "compliance_principle": "No Evidence, No Compliance Claim",
        "rag_enabled": _rag_feature_enabled(),
        "rag_citations": rag_citations,
        "llm_model": _current_llm_info(),
    })


def _offline_vendor_status():
    """確認前端離線套件是否齊全（Chart / Mermaid / PDF）。"""
    vendor = Path(BASE_DIR) / "static" / "vendor"
    required = [
        "chart.umd.min.js",
        "mermaid.min.js",
        "jspdf.umd.min.js",
        "html2canvas.min.js",
    ]
    missing = [name for name in required if not (vendor / name).is_file()]
    return {
        "vendor_dir": str(vendor),
        "ready": not missing,
        "missing": missing,
        "hf_offline": os.environ.get("HF_HUB_OFFLINE", "") in ("1", "true", "True"),
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE", "") in ("1", "true", "True"),
    }


def _path_has_causal_weights(path: Path) -> bool:
    """是否有可直接 from_pretrained 的權重（略過純 LoRA adapter）。"""
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    if any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin")):
        return True
    if any(path.glob("model*.safetensors")) or any(path.glob("model*.bin")):
        return True
    # 分片索引
    if (path / "model.safetensors.index.json").is_file():
        return True
    if (path / "pytorch_model.bin.index.json").is_file():
        return True
    return False


def _load_model_presets() -> dict:
    presets_path = Path(BASE_DIR) / "train_ai" / "train_llm" / "model_presets.json"
    if not presets_path.is_file():
        return {}
    try:
        with open(presets_path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        return data.get("presets") or {}
    except Exception:
        return {}


def _same_llm_ref(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if _is_hub_model_id(a) or _is_hub_model_id(b):
        return a.strip() == b.strip()
    try:
        return Path(a).resolve() == Path(b).resolve()
    except Exception:
        return a == b


def _infer_stage(slug: str, path: str, meta: dict | None = None) -> str:
    """finetuned=微調後；base=微調前。"""
    meta = meta or {}
    if meta.get("stage") in ("base", "finetuned"):
        return meta["stage"]
    if _is_hub_model_id(path) or _is_hub_model_id(slug) or str(slug).startswith("base:"):
        return "base"
    name = (slug or Path(path).name or "").lower()
    if any(k in name for k in ("merged", "_ot", "finetune", "lora")):
        return "finetuned"
    if meta.get("base_model_id"):
        return "finetuned"
    return "finetuned"


def _hf_hub_roots() -> list[Path]:
    roots = []
    for key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE"):
        v = (os.environ.get(key) or "").strip()
        if v:
            roots.append(Path(v))
    roots.append(Path.home() / ".cache" / "huggingface")
    # HF_HOME 本身或 hub 子目錄
    out = []
    for r in roots:
        out.append(r)
        out.append(r / "hub")
    # 去重
    seen = set()
    uniq = []
    for r in out:
        key = str(r)
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    return uniq


def _find_hf_snapshot(model_id: str) -> str | None:
    """在本機 HuggingFace cache 找 snapshot 目錄（不連網）。"""
    if not _is_hub_model_id(model_id):
        return None
    repo = "models--" + model_id.replace("/", "--")
    for root in _hf_hub_roots():
        snaps = root / repo / "snapshots"
        if not snaps.is_dir():
            # root 可能已是 hub
            alt = root / "hub" / repo / "snapshots"
            snaps = alt if alt.is_dir() else snaps
        if not snaps.is_dir():
            continue
        # 優先有 config.json 的 snapshot（依 mtime 新到舊）
        cands = sorted(
            [p for p in snaps.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for snap in cands:
            if (snap / "config.json").is_file():
                return str(snap.resolve())
    return None


def _hub_id_from_local_path(path: str) -> str | None:
    """
    從 HF cache 路徑反推 hub id。
    例：.../models--Qwen--Qwen2.5-3B-Instruct/snapshots/<hash> → Qwen/Qwen2.5-3B-Instruct
    """
    return _early_hub_id_from_path(path)


def _resolve_snapshot_hash(hash_or_path: str) -> str | None:
    """snapshot 雜湊或路徑 → 可載入的本機 snapshot 目錄。"""
    raw = (hash_or_path or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if p.is_dir() and (p / "config.json").is_file():
        return str(p.resolve())
    # 純 hash：在所有 hub cache 裡找
    if re.fullmatch(r"[0-9a-f]{7,64}", raw, re.I):
        for root in _hf_hub_roots():
            try:
                for snap in root.glob(f"models--*/snapshots/{raw}"):
                    if snap.is_dir() and (snap / "config.json").is_file():
                        return str(snap.resolve())
            except Exception:
                continue
            # root 已是 hub
            try:
                for snap in (root / "hub").glob(f"models--*/snapshots/{raw}"):
                    if snap.is_dir() and (snap / "config.json").is_file():
                        return str(snap.resolve())
            except Exception:
                continue
    return None


def _hub_model_load_ref(model_id: str) -> tuple[str | None, str | None]:
    """
    回傳 (load_path, error)。
    優先本機 cache snapshot；無快取且不允許連網則明確錯誤（不連 HF）。
    """
    # 已是本機目錄
    p = Path(model_id)
    if p.is_dir() and (p / "config.json").is_file():
        return str(p.resolve()), None

    snap = _find_hf_snapshot(model_id)
    if snap:
        return snap, None

    offline = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in (
        "1", "true", "yes"
    ) or os.environ.get("TRANSFORMERS_OFFLINE", "").strip().lower() in (
        "1", "true", "yes"
    )
    allow_dl = _env_flag("ALLOW_HF_DOWNLOAD", default=False)
    if offline or not allow_dl:
        # 建議改選本機已有快取
        cached_hints = []
        for mid in (
            "Qwen/Qwen2.5-3B-Instruct",
            "microsoft/Phi-4-mini-instruct",
            "Qwen/Qwen3-4B-Instruct-2507",
        ):
            if _find_hf_snapshot(mid):
                cached_hints.append(mid)
        hint = (
            ("本機已有快取可選：" + "、".join(cached_hints))
            if cached_hints
            else "請改選「微調後」本地模型，或先連網下載"
        )
        return None, (
            f"微調前模型「{model_id}」本機沒有 HuggingFace 快取，"
            f"且目前禁止連網下載（離線／未設 ALLOW_HF_DOWNLOAD=1）。{hint}。"
        )
    # 明確允許下載時才回 hub id
    return model_id, None


def _discover_local_llm_models() -> list[dict]:
    """掃描微調後本地模型 + 微調前基底（presets／對應 base_model_id）。"""
    if USE_OLLAMA:
        return ollama_service.list_models_for_ui()

    root = Path(BASE_DIR)
    train_ai = root / "train_ai"
    train_llm = train_ai / "train_llm"
    models_dir = train_ai / "models"
    items: list[dict] = []
    seen: set[str] = set()
    base_ids: set[str] = set()

    def add_item(
        slug: str,
        path: str,
        *,
        stage: str,
        base_model_id: str | None = None,
        desc: str = "",
        available: bool = True,
        unavailable_reason: str = "",
    ):
        key = f"{stage}:{path}"
        if key in seen:
            return
        seen.add(key)
        cur = _normalize_llm_ref(MODEL_PATH) if MODEL_PATH else ""
        cur_hub = _hub_id_from_local_path(cur) if cur else None
        load_path = path
        if stage == "base" and _is_hub_model_id(path):
            ref, err = _hub_model_load_ref(path)
            if err:
                available = False
                unavailable_reason = err
            elif ref:
                load_path = ref
                if ref != path:
                    desc = (desc + " · 本機快取就緒").strip(" ·")
        # 目前載入是否為 HF 基底（snapshot／hub id）——不可拿來標「微調後」使用中
        cur_is_base = bool(cur_hub) or _is_hub_model_id(cur)
        if stage == "base":
            hub_id = path if _is_hub_model_id(path) else (base_model_id or "")
            active = bool(
                cur_is_base
                and (
                    (cur_hub and hub_id and cur_hub == hub_id)
                    or (cur_hub and slug == f"base:{cur_hub}")
                    or _same_llm_ref(load_path, cur)
                    or _same_llm_ref(path, cur)
                )
            )
        else:
            # 微調後：必須對到本地路徑／slug；禁止用 base_model_id 對上微調前快取
            if cur_is_base:
                active = False
            else:
                cur_name = Path(cur).name if cur else ""
                active = bool(
                    _same_llm_ref(path, cur)
                    or _same_llm_ref(load_path, cur)
                    or (slug and slug == cur_name)
                    or (slug and cur.endswith(slug))
                )
        items.append({
            "slug": slug,
            "label": _llm_friendly_name(slug if stage == "finetuned" else path),
            "path": path,
            "load_path": load_path,
            "stage": stage,
            "stage_label": "微調後" if stage == "finetuned" else "微調前",
            "base_model_id": base_model_id,
            "desc": desc,
            "source": "huggingface" if _is_hub_model_id(path) else "local",
            "available": available,
            "unavailable_reason": unavailable_reason,
            "active": active,
        })

    def add_finetuned(slug: str, path: Path):
        if not _path_has_causal_weights(path):
            return
        meta = {}
        meta_path = path / "train_meta.json"
        if meta_path.is_file():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f) or {}
            except Exception:
                meta = {}
        base_id = meta.get("base_model_id") or _FINETUNED_BASE_FALLBACK.get(slug)
        if base_id:
            base_ids.add(base_id)
        available = True
        unavailable_reason = ""
        desc = "本地微調合併模型" + (f" ← {base_id}" if base_id else "")
        if _model_is_prequantized(str(path.resolve())):
            desc += " · CUDA 4-bit"
        elif not _model_full_weights_ok(str(path.resolve())):
            adapter = _lora_adapter_dir(path)
            if adapter is not None and base_id:
                desc = f"merge 不完整 → 自動用基底+LoRA（{base_id}）"
            else:
                available = False
                unavailable_reason = (
                    "權重不完整（會 MISMATCH）。請重新 merge，"
                    f"或改選基底 {base_id or 'Instruct'}。"
                )
        add_item(
            slug,
            str(path.resolve()),
            stage="finetuned",
            base_model_id=base_id,
            desc=desc,
            available=available,
            unavailable_reason=unavailable_reason,
        )

    for base in (models_dir, train_llm):
        if not base.is_dir():
            continue
        for p in sorted(base.iterdir()):
            if p.is_dir() and (p / "config.json").is_file():
                add_finetuned(p.name, p)

    for slug, cands in {
        "qwen_ot_merged_model": [
            train_llm / "qwen_ot_merged_model",
            root / "qwen_ot_merged_model",
            train_ai / "qwen_ot_merged_model",
        ],
        "phi4_merged_model": [
            train_llm / "phi4_merged_model",
            train_ai / "phi4_merged_model",
            root / "phi4_merged_model",
        ],
    }.items():
        for c in cands:
            if (c / "config.json").is_file():
                add_finetuned(slug, c)
                break

    # 微調前：presets + 微調成品對應的基底
    presets = _load_model_presets()
    for key, meta in presets.items():
        mid = (meta or {}).get("model_id")
        if mid:
            base_ids.add(mid)
    for mid in sorted(base_ids):
        if not _is_hub_model_id(mid):
            continue
        preset_desc = ""
        for meta in presets.values():
            if (meta or {}).get("model_id") == mid:
                preset_desc = (meta or {}).get("desc") or ""
                break
        add_item(
            f"base:{mid}",
            mid,
            stage="base",
            base_model_id=mid,
            desc=preset_desc or "原廠／基底 Instruct（微調前）",
        )

    # 微調後在前、微調前在後（前端也會分組）
    items.sort(key=lambda x: (0 if x["stage"] == "finetuned" else 1, x["label"]))
    return items


def _current_llm_info() -> dict:
    if USE_OLLAMA:
        info = ollama_service.current_info()
        info["speed_mode"] = SPEED_MODE
        info["edge_mode"] = bool(EDGE_MODE)
        return info

    cur = _normalize_llm_ref(MODEL_PATH) if MODEL_PATH else ""
    hub_from_path = _hub_id_from_local_path(cur) if cur else None
    if _is_hub_model_id(cur):
        slug = f"base:{cur}"
        stage = "base"
        label_key = cur
        display_path = cur
    elif hub_from_path:
        # HF snapshot 目錄：不要用 hash 當 slug（前端切換會失敗）
        slug = f"base:{hub_from_path}"
        stage = "base"
        label_key = hub_from_path
        display_path = hub_from_path
    else:
        slug = Path(cur).name if cur else ""
        stage = _infer_stage(slug, cur)
        label_key = slug
        display_path = cur
    try:
        dev = str(next(model.parameters()).device) if model is not None else "n/a"
    except Exception:
        dev = "n/a"
    return {
        "slug": slug,
        "label": _llm_friendly_name(label_key) if label_key else "",
        "path": display_path,
        "load_path": cur,
        "stage": stage,
        "stage_label": "微調後" if stage == "finetuned" else "微調前",
        "loaded": model is not None and tokenizer is not None,
        "edge_mode": bool(EDGE_MODE),
        "device": LLM_DEVICE,
        "runtime_device": dev,
        "speed_mode": SPEED_MODE,
    }


def _resolve_switch_target(target: str) -> dict:
    """
    解析切換目標。注意：不可用 finetuned.base_model_id 去匹配「微調前」請求，
    否則選基底會誤載入微調後模型。
    """
    raw = (target or "").strip()
    if not raw:
        return {"ok": False, "error": "請指定 model slug 或路徑"}

    items = _discover_local_llm_models()
    want_base = raw.startswith("base:")
    hub_or_raw = raw[5:].strip() if want_base else raw

    # 0) HF snapshot 雜湊或 snapshot 路徑 → 還原成 hub 模型
    snap_path = _resolve_snapshot_hash(raw) or _resolve_snapshot_hash(hub_or_raw)
    hub_from_raw = _hub_id_from_local_path(raw) or _hub_id_from_local_path(hub_or_raw)
    if snap_path or hub_from_raw:
        hub_id = hub_from_raw or _hub_id_from_local_path(snap_path or "")
        load_path = snap_path
        if hub_id and not load_path:
            load_path, err = _hub_model_load_ref(hub_id)
            if err:
                return {"ok": False, "error": err, "slug": f"base:{hub_id}"}
        if hub_id and load_path:
            # 若列表裡已有該 base，沿用其 metadata
            for item in items:
                if item.get("stage") == "base" and hub_id in (
                    item.get("path"),
                    item.get("base_model_id"),
                    item.get("slug", "").removeprefix("base:"),
                ):
                    item = dict(item)
                    item["load_path"] = load_path
                    return {"ok": True, "item": item}
            return {
                "ok": True,
                "item": {
                    "slug": f"base:{hub_id}",
                    "path": hub_id,
                    "load_path": load_path,
                    "stage": "base",
                    "available": True,
                },
            }

    # 1) 精確 slug（含 base:xxx）
    for item in items:
        if raw == item.get("slug"):
            return {"ok": True, "item": item}
    # 2) 精確 path / load_path（含 snapshot 完整路徑）
    for item in items:
        if raw == item.get("path") or raw == item.get("load_path"):
            return {"ok": True, "item": item}
        if item.get("load_path") and _same_llm_ref(raw, item.get("load_path") or ""):
            return {"ok": True, "item": item}
    # 3) 明確要微調前：只在 stage=base 裡用 hub id 匹配
    if want_base or _is_hub_model_id(hub_or_raw):
        for item in items:
            if item.get("stage") != "base":
                continue
            if hub_or_raw in (
                item.get("path"),
                item.get("base_model_id"),
                item.get("slug", "").removeprefix("base:"),
            ):
                return {"ok": True, "item": item}
        if _is_hub_model_id(hub_or_raw):
            load_ref, err = _hub_model_load_ref(hub_or_raw)
            if err:
                return {"ok": False, "error": err, "slug": f"base:{hub_or_raw}"}
            return {
                "ok": True,
                "item": {
                    "slug": f"base:{hub_or_raw}",
                    "path": hub_or_raw,
                    "load_path": load_ref or hub_or_raw,
                    "stage": "base",
                    "available": True,
                },
            }
    # 4) 本地目錄
    if not _is_hub_model_id(raw):
        for item in items:
            if item.get("stage") == "finetuned" and _same_llm_ref(raw, item.get("path") or ""):
                return {"ok": True, "item": item}
        p = Path(raw)
        if _path_has_causal_weights(p):
            return {
                "ok": True,
                "item": {
                    "slug": p.name,
                    "path": str(p.resolve()),
                    "load_path": str(p.resolve()),
                    "stage": "finetuned",
                    "available": True,
                },
            }
    return {"ok": False, "error": f"找不到可載入模型：{raw}"}


def switch_llm_model(target: str) -> dict:
    """依 slug／路徑／HF id 切換 LLM；已載入者走快取，不重複載入。"""
    global model, tokenizer, MODEL_PATH, _llm_chat_primed

    print(f"📥 LLM switch 請求：{target!r}")
    _reset_chat_template_probe()

    if USE_OLLAMA:
        result = ollama_service.switch_model(target)
        if not result.get("ok"):
            print(f"❌ Ollama switch 失敗：{result.get('error')}")
            return result
        MODEL_PATH = ollama_service.current_model_name()
        _llm_chat_primed = False
        try:
            _prime_llm_chat(force=True)
        except Exception as we:
            print(f"⚠️ Ollama 切換後對話探測略過：{we}")
        info = _current_llm_info()
        if result.get("slug"):
            info["slug"] = result["slug"]
        if result.get("label"):
            info["label"] = result["label"]
        return {
            **result,
            **info,
        }

    resolved_meta = _resolve_switch_target(target)
    if not resolved_meta.get("ok"):
        print(f"❌ LLM switch 無法解析：{resolved_meta.get('error')}")
        return resolved_meta

    item = resolved_meta["item"]
    slug = item.get("slug") or ""
    stage = item.get("stage") or "finetuned"
    hub_id = item.get("path") if stage == "base" else None

    if stage == "base" and _is_hub_model_id(item.get("path") or ""):
        load_ref, err = _hub_model_load_ref(item["path"])
        if err:
            print(f"❌ LLM switch 微調前不可用：{err}")
            return {"ok": False, "error": err, "slug": slug, "stage": stage}
        resolved = _normalize_llm_ref(load_ref)
    else:
        resolved = _normalize_llm_ref(item.get("load_path") or item.get("path") or "")

    if not resolved:
        return {"ok": False, "error": f"無法解析載入路徑：{target}"}

    # CPU 載不了 4-bit → 自動改試 GPU；不完整 merge 不靠 GPU（一樣 MISMATCH）
    gpu_fallback_used = False
    if not _use_cuda_for_llm():
        need_gpu = False
        why = ""
        if _model_is_prequantized(resolved):
            need_gpu = True
            why = f"「{slug or Path(resolved).name}」為 CUDA 4-bit，CPU 無法載入"
        elif (
            Path(resolved).is_dir()
            and (not _model_full_weights_ok(resolved))
            and (_lora_adapter_dir(resolved) is None)
        ):
            # 無 LoRA 可救 → 直接拒絕（勿誤當 GPU 可解）
            return {
                "ok": False,
                "error": (
                    f"「{slug or Path(resolved).name}」權重不完整，載入會 MISMATCH。"
                    "請重新完整 merge，或改選對應基底 Instruct／其他完整模型。"
                ),
                "slug": slug,
            }
        if need_gpu:
            if _try_enable_cuda_for_model(why):
                gpu_fallback_used = True
            else:
                return {
                    "ok": False,
                    "error": (
                        f"{why}，且目前無法改用 GPU。"
                        "請改選本機快取的 Qwen2.5-3B-Instruct／Phi-4-mini，"
                        "或設 LLM_ALLOW_GPU_FALLBACK=1 後重啟再試微調模型。"
                    ),
                    "slug": slug,
                }

    display_path = hub_id if (stage == "base" and hub_id) else resolved
    cache_key = resolved

    cur_info = _current_llm_info()
    already_active = False
    if model is not None and tokenizer is not None:
        if any(
            e.get("model") is model and e.get("cache_key") == cache_key
            for e in _llm_cache.values()
        ):
            already_active = True
        elif _same_llm_ref(display_path, MODEL_PATH or ""):
            already_active = True
        elif cur_info.get("slug") and cur_info.get("slug") == slug:
            already_active = True
    if already_active:
        with _llm_lock:
            if cache_key in _llm_cache:
                _llm_cache_touch(cache_key)
            elif model is not None and tokenizer is not None:
                # 作用中但尚未入快取（理論少見）
                _llm_cache_put(
                    cache_key, model, tokenizer,
                    display_path=display_path,
                    slug=slug or Path(str(display_path)).name,
                    stage=stage,
                )
            if _LLM_RELEASE_ON_SWITCH:
                _llm_cache_release_except(cache_key)
        return {
            "ok": True,
            "switched": False,
            "from_cache": True,
            "message": "已是目前使用的模型",
            **_current_llm_info(),
            "cached_models": _llm_cache_keys_info(),
        }

    with _llm_lock:
        if _llm_cache_activate(cache_key):
            if _LLM_RELEASE_ON_SWITCH:
                _llm_cache_release_except(cache_key)
            print(f"⚡ 使用快取模型：{slug or cache_key}")
            _llm_chat_primed = False
            try:
                _prime_llm_chat(force=True)
            except Exception as we:
                print(f"⚠️ 快取切換後對話探測略過：{we}")
            return {
                "ok": True,
                "switched": True,
                "from_cache": True,
                "message": (
                    f"已切換為 {_current_llm_info().get('label') or slug}"
                    "（快取命中；其他模型已釋放）"
                ),
                **_current_llm_info(),
                "cached_models": _llm_cache_keys_info(),
            }

        prev_path = MODEL_PATH
        prev_key = None
        for k, e in _llm_cache.items():
            if e.get("model") is model:
                prev_key = k
                break

        print(f"🔄 載入 LLM：{resolved}（{stage}）；將釋放舊模型資源")
        try:
            # 載入前先卸載舊模型，避免雙模型同時佔顯存
            if _LLM_RELEASE_ON_SWITCH:
                _llm_cache_release_except(None)
                model, tokenizer = None, None
            elif prev_key:
                _llm_cache_evict_if_needed(keep_key=prev_key)

            new_m, new_t = _load_llm(resolved)
            if new_m is None or new_t is None:
                raise RuntimeError("模型載入回傳空值")

            MODEL_PATH = display_path
            model, tokenizer = new_m, new_t
            _llm_cache_put(
                cache_key,
                new_m,
                new_t,
                display_path=display_path,
                slug=slug or Path(str(display_path)).name,
                stage=stage,
            )
            if _LLM_RELEASE_ON_SWITCH:
                _llm_cache_release_except(cache_key)

            # 切換後重做對話探測（新模型同樣會有冷啟動第一則不穩）
            _llm_chat_primed = False
            try:
                _prime_llm_chat(force=True)
            except Exception as we:
                print(f"⚠️ 切換後對話探測略過：{we}")
                if _is_cuda_fault(we):
                    print("⚠️ 切換後探測 CUDA 崩潰 → 切 CPU 後備")
                    _failover_llm_to_cpu(str(we))

            info = _current_llm_info()
            msg = f"已載入並切換為 {info.get('label') or slug}（舊模型資源已釋放）"
            if gpu_fallback_used:
                msg += " · 已自動改以 GPU 載入"
            print(f"✅ LLM 已載入並切換為 {info.get('label') or slug}（舊模型已釋放）")
            return {
                "ok": True,
                "switched": True,
                "from_cache": False,
                "gpu_fallback": gpu_fallback_used,
                "message": msg,
                **info,
                "cached_models": _llm_cache_keys_info(),
            }
        except Exception as e:
            print(f"❌ LLM 切換失敗，嘗試回復：{e}")
            # GPU 後備也失敗且為 CUDA 故障 → 退回 CPU，避免整程掛掉
            if gpu_fallback_used and _is_cuda_fault(e):
                try:
                    _failover_llm_to_cpu(str(e))
                except Exception:
                    pass
            MODEL_PATH = prev_path
            try:
                prev_ck = _normalize_llm_ref(prev_path) if prev_path else ""
                if prev_ck and _llm_cache_activate(prev_ck):
                    pass
                elif prev_path:
                    model, tokenizer = _load_llm(prev_path)
                    _llm_cache_put(
                        prev_ck or _normalize_llm_ref(prev_path),
                        model,
                        tokenizer,
                        display_path=prev_path,
                        slug=Path(str(prev_path)).name,
                        stage="finetuned",
                    )
                    if _LLM_RELEASE_ON_SWITCH:
                        _llm_cache_release_except(
                            prev_ck or _normalize_llm_ref(prev_path)
                        )
                else:
                    raise RuntimeError("無先前模型可回復")
            except Exception as e2:
                model, tokenizer = None, None
                return {
                    "ok": False,
                    "error": f"切換失敗且無法回復舊模型：{e}／{e2}",
                    "path": prev_path,
                }
            return {
                "ok": False,
                "error": f"切換失敗，已回復原模型：{e}",
                **_current_llm_info(),
                "cached_models": _llm_cache_keys_info(),
            }



@app.route('/api/llm/models', methods=['GET'])
def list_llm_models():
    """列出微調前（基底）與微調後（本地）可切換模型。"""
    models = _discover_local_llm_models()
    groups = {
        "finetuned": [m for m in models if m.get("stage") in ("finetuned", "alias")],
        "base": [m for m in models if m.get("stage") == "base"],
    }
    return jsonify({
        "current": _current_llm_info(),
        "models": models,
        "groups": groups,
        "cached_models": _llm_cache_keys_info(),
        "cache_max": _LLM_CACHE_MAX,
        "release_on_switch": _LLM_RELEASE_ON_SWITCH,
        "hint": (
            "Ollama 後端：切換即時生效。微調後＝Semi-Shield 別名／本地 merge；微調前＝官方基底。"
            if USE_OLLAMA
            else (
                "切換模型後會釋放舊模型顯存／記憶體，再載入新模型。"
                "微調後＝本地 merged；微調前＝HF 基底（需本機 cache）。"
            )
        ),
    })



@app.route('/api/llm/switch', methods=['POST'])
def api_switch_llm_model():
    """熱切換對話用 LLM（卸載舊模型後載入選定模型）。"""
    req = request.get_json(silent=True) or {}
    target = (req.get("model") or req.get("slug") or req.get("path") or "").strip()
    print(f"🌐 /api/llm/switch body keys={list(req.keys())} model={target!r}")
    if not target:
        return jsonify({"ok": False, "error": "請傳入 JSON：{'model': '模型 slug 或路徑'}"}), 400
    result = switch_llm_model(target)
    # 業務失敗仍回 200 + ok:false，避免前端只看到籠統 400
    if not result.get("ok"):
        return jsonify(result), 200
    return jsonify(result), 200


@app.route('/api/safety/status', methods=['GET'])
def safety_status():
    """回傳 RAG / 護欄 / 離線資源就緒狀態，方便前端顯示。"""
    offline = _offline_vendor_status()
    rs = _get_rag_service()
    return jsonify({
        "rag_enabled": _rag_feature_enabled(),
        "rag_docs": len(getattr(rs, "docs", []) or []) if rs else 0,
        "rag_mode": getattr(rs, "mode", "off") if rs else "off",
        "guardrail_mode": "disabled" if not ENABLE_GUARDRAIL else guardrail_service.mode,
        "guardrail_enabled": ENABLE_GUARDRAIL,
        "guardrail_mechanism": (
            guardrail_service.mechanism_summary()
            if hasattr(guardrail_service, "mechanism_summary")
            else {"mode": "disabled", "input_layers": [], "output_layers": [], "human_review": []}
        ),
        "guardrail_model_dir": str(
            (Path(BASE_DIR) / "train_ai" / "train_gur" / "fine_tuned_guardrail")
        ),
        "compliance_principle": "No Evidence, No Compliance Claim",
        "compliance_coverage": coverage_stats(),
        "evidence_traceability": traceability_stats(),
        "human_review": review_stats(),
        "reviewer_mode": reviewer_mode_summary(),
        "offline_ready": offline["ready"],
        "offline": offline,
        "llm_model_path": MODEL_PATH,
        "llm_model": _current_llm_info(),
        "llm_models": _discover_local_llm_models(),
        "edge_mode": bool(EDGE_MODE),
        "llm_device": LLM_DEVICE,
        "speed_mode": SPEED_MODE,
        "cuda_available": _torch_cuda_available(),
    })


@app.route('/compliance')
def serve_compliance_page():
    """合規架構、GRC 對標、Agent Workflow、KPI 說明頁。"""
    return send_from_directory(WEB_DIR, 'compliance.html')


@app.route('/api/compliance/metrics', methods=['GET'])
def api_compliance_metrics():
    """各項 KPI／覆蓋率／Evidence／監控指標詳細分析。"""
    limit = min(int(request.args.get("limit") or 50), 200)
    evidence_payload = list_evidence(limit=limit)
    return jsonify(build_metrics_analysis(
        traceability=traceability_stats(),
        human_review=review_stats(),
        evidence_items=evidence_payload,
    ))


@app.route('/api/compliance/matrix', methods=['GET'])
def api_compliance_matrix():
    """ISO 27001:2022 Annex A 矩陣與覆蓋率。"""
    phase1_only = str(request.args.get("phase1") or "").strip().lower() in ("1", "true", "yes")
    controls = load_controls()
    if phase1_only:
        controls = [c for c in controls if c.get("phase1_mvp")]
    return jsonify({
        "principle": "No Evidence, No Compliance Claim",
        "coverage": coverage_stats(),
        "coverage_gaps": coverage_gaps(),
        "controls": controls,
        "kpi_targets": load_compliance_json("kpi_targets.json"),
    })


@app.route('/api/compliance/grc', methods=['GET'])
def api_compliance_grc():
    """商業 GRC 方案對標與差異化定位。"""
    return jsonify(load_compliance_json("grc_positioning.json"))


@app.route('/api/compliance/workflow', methods=['GET'])
def api_compliance_workflow():
    """Agent Workflow 四角色管線說明。"""
    return jsonify(workflow_spec())


@app.route('/api/compliance/pipeline', methods=['POST'])
def api_compliance_pipeline():
    """執行 Collector → Auditor → Reviewer → Reporter 合規管線（地端）。"""

    def _scan():
        return get_ot_monitor_data(force=True)

    def _audit(key: str, bundle: dict):
        log_content = bundle.get("log") or ""
        title = bundle.get("title") or key
        metric_summary = bundle.get("metric_summary") or ""
        rag_context, rag_citations = "", []
        if needs_rag_for_audit(str(log_content), title):
            rq = f"{title} ISO 27001 {str(log_content)[:240]}"
            rag_context, rag_citations, _ = retrieve_rag_for_query(rq, force=True)
        analysis = ask_llm(
            log_content,
            control_key=key,
            title=title,
            metric_summary=metric_summary,
            rag_context=rag_context or None,
        )
        return {"ai_analysis": analysis, "rag_citations": rag_citations}

    result = run_compliance_pipeline(
        scan_fn=_scan,
        audit_fn=_audit,
        enqueue_fn=enqueue_review,
        evidence_fn=register_control_bundle_evidence,
    )
    if ai_reviewer_enabled():
        result["ai_review"] = run_ai_reviewer()
    result["compliance_coverage"] = coverage_stats()
    result["human_review"] = review_stats()
    result["reviewer_mode"] = reviewer_mode_summary()
    return jsonify(result)


@app.route('/api/evidence', methods=['GET'])
def api_list_evidence():
    limit = min(500, int(request.args.get("limit") or 100))
    control_key = (request.args.get("control_key") or "").strip() or None
    return jsonify({
        "principle": "No Evidence, No Compliance Claim",
        "schema": load_compliance_json("evidence_schema.json"),
        "items": list_evidence(limit=limit, control_key=control_key),
        "traceability": traceability_stats(),
    })


@app.route('/api/evidence/<evidence_id>', methods=['GET'])
def api_get_evidence(evidence_id):
    rec = get_evidence(evidence_id)
    if not rec:
        return jsonify({"error": "找不到 evidence_id"}), 404
    return jsonify(rec)


@app.route('/api/review/queue', methods=['GET'])
def api_review_queue():
    status = (request.args.get("status") or "").strip() or None
    return jsonify({
        "stats": review_stats(),
        "items": list_reviews(status=status),
    })


@app.route('/api/review/resolve', methods=['POST'])
def api_review_resolve():
    req = request.get_json(silent=True) or {}
    review_id = (req.get("review_id") or "").strip()
    status = (req.get("status") or "").strip().lower()
    if not review_id or status not in ("approved", "rejected"):
        return jsonify({"error": "需要 review_id 與 status=approved|rejected"}), 400
    rec = resolve_review(
        review_id,
        status=status,
        reviewer=(req.get("reviewer") or "analyst").strip(),
        notes=(req.get("notes") or "").strip(),
    )
    if not rec:
        return jsonify({"error": "找不到 review_id"}), 404
    return jsonify({"ok": True, "item": rec, "stats": review_stats()})


@app.route('/api/review/auto', methods=['POST'])
def api_review_auto():
    """無分析師時：AI Reviewer 批次審核 pending queue + evidence。"""
    body = request.get_json(silent=True) or {}
    result = run_ai_reviewer(
        limit=min(int(body.get("limit") or 500), 2000),
        reconcile=body.get("reconcile", True) is not False,
    )
    if not result.get("ok"):
        return jsonify(result), 400
    return jsonify(result)


@app.route('/api/review/mode', methods=['GET'])
def api_review_mode():
    return jsonify(reviewer_mode_summary())


def _lan_ipv4_addrs() -> list[str]:
    """取得本機區網／VPN IPv4（含 Radmin VPN 的 26.x）。"""
    import socket
    import subprocess

    addrs: list[str] = []

    def _add(ip: str):
        if not ip or ip.startswith("127.") or ip in addrs:
            return
        if ip.startswith("169.254."):
            return
        addrs.append(ip)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            _add(s.getsockname()[0])
    except Exception:
        pass
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            _add(info[4][0])
    except Exception:
        pass
    # Windows：用 ipconfig 抓所有 IPv4（含 Radmin VPN / Hamachi）
    if os.name == "nt":
        try:
            out = subprocess.check_output(
                ["ipconfig"], text=True, encoding="utf-8", errors="ignore"
            )
            for line in out.splitlines():
                if "IPv4" in line or "IP Address" in line:
                    part = line.split(":")[-1].strip()
                    # 去掉可能的 (Preferred) 後綴
                    ip = part.split("(")[0].strip()
                    _add(ip)
        except Exception:
            pass
    # VPN 網段優先顯示
    addrs.sort(
        key=lambda ip: (
            0 if ip.startswith("26.") else (1 if ip.startswith("25.") else 2),
            ip,
        )
    )
    return addrs


# =======================================================
# 4. 伺服器啟動入口
# =======================================================
def run_server() -> None:
    if not os.path.exists(OT_FOLDER):
        os.makedirs(OT_FOLDER)
    offline = _offline_vendor_status()
    port = int(os.environ.get("PORT", "2000"))
    lan_ips = _lan_ipv4_addrs()
    print("\n🚀 Semi-Shield 智慧網路資安與 LLM 整合後端伺服器已啟動。")
    print("🏠 本機：http://127.0.0.1:%d/  或  http://127.0.0.1:%d/platform" % (port, port))
    print("💬 聊天：http://127.0.0.1:%d/chat" % port)
    print("📊 監控：http://127.0.0.1:%d/monitor" % port)
    if lan_ips:
        print("🌐 區網（其他電腦請用同一 Wi‑Fi／內網開啟）：")
        for ip in lan_ips:
            print(f"   http://{ip}:{port}/")
            print(f"   http://{ip}:{port}/chat")
            print(f"   http://{ip}:{port}/monitor")
    else:
        print("🌐 區網：無法自動偵測 IP，請用 ipconfig 查看後連 http://<你的IP>:%d/" % port)
    print("   （若連不上：Windows 防火牆需允許 Python／埠 %d 傳入）" % port)
    print(f"📚 RAG：{'啟用' if _rag_feature_enabled() else '停用（功能關閉）'}")
    print(f"🛡️ 護欄：{'啟用 (' + guardrail_service.mode + ')' if ENABLE_GUARDRAIL else '停用'}")
    print(
        f"📴 離線前端套件：{'就緒' if offline['ready'] else '缺少 ' + ','.join(offline['missing'])}"
    )
    if EDGE_MODE:
        print(
            "🍊 Edge 提示：預設 Qwen2.5-0.5B + 關 RAG/護欄；"
            "8GB 記憶體可設 EDGE_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct"
        )

    def _warm_ot_monitor_cache():
        try:
            get_ot_monitor_data(force=True)
            print("📦 OT 監控快取預熱完成（後續輪詢不再重讀檔）")
        except Exception as e:
            print(f"⚠️ OT 監控快取預熱略過：{e}")

    threading.Thread(target=_warm_ot_monitor_cache, daemon=True).start()
    _threaded = not EDGE_MODE
    if os.environ.get("FLASK_THREADED", "").strip() != "":
        _threaded = _env_flag("FLASK_THREADED", default=_threaded)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=_threaded)


if __name__ == "__main__":
    run_server()