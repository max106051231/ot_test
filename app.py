import os
import re
import json
import torch
import gc
from pathlib import Path

# 離線優先：本地模型目錄存在時，禁止 Transformers / HF 連外網抓檔
_BASE_EARLY = Path(__file__).resolve().parent


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


if _local_llm_dirs_exist(_BASE_EARLY):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from guardrail_service import guardrail_service
from rag_service import rag_service

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "static"), static_url_path="/static")
CORS(app)

# =======================================================
# 護欄開關（False = 略過輸入攔截 / 輸出脫敏；True = 啟用）
# =======================================================
ENABLE_GUARDRAIL = True


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


if not ENABLE_GUARDRAIL:
    guardrail_service = _DisabledGuardrail()
    print("🛡️ 護欄功能已停用（ENABLE_GUARDRAIL=False）")

# =======================================================
# 全局變數與配置（速度優先；顯存不足時可設 LLM_4BIT=1）
# =======================================================
# 一律相對專案根目錄（不依賴啟動 cwd）
_ot_dir = Path(BASE_DIR) / "ot"
_OT_dir = Path(BASE_DIR) / "OT"
OT_FOLDER = str(_ot_dir if _ot_dir.is_dir() else (_OT_dir if _OT_dir.is_dir() else _ot_dir))


def _llm_candidate_dirs(root: Path) -> list[Path]:
    """依優先序回傳可能的本地 LLM 目錄。"""
    train_ai = root / "train_ai"
    train_llm = train_ai / "train_llm"
    models_dir = train_ai / "models"
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


def _resolve_llm_model_path() -> str:
    """
    解析推論模型路徑（優先序）：
      1) 環境變數 LLM_MODEL_PATH
      2) train_ai/models/<slug>/（新版訓練輸出）
      3) train_ai/train_llm/*_merged_model（目前實際存放處）
      4) 專案根 / train_ai 頂層舊路徑
    """
    env = (os.environ.get("LLM_MODEL_PATH") or "").strip()
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = Path(BASE_DIR) / p
        if (p / "config.json").is_file():
            return str(p.resolve())
        print(f"⚠️ LLM_MODEL_PATH 無效或不含 config.json：{env}")

    root = Path(BASE_DIR)
    for p in _llm_candidate_dirs(root):
        if (p / "config.json").is_file():
            return str(p.resolve())
    return str((root / "train_ai" / "train_llm" / "qwen_ot_merged_model").resolve())


MODEL_PATH = _resolve_llm_model_path()
print(f"🧠 LLM 模型路徑：{MODEL_PATH}")
print(f"📁 OT 日誌目錄：{OT_FOLDER}")

# 速度：生成長度 ≈ 延遲；結束標記會提早停止，不必給過大上限
MAX_INPUT_CHARS = 800       # Log/摘要最大字元（留給完整回答）
MAX_NEW_TOKENS = 520        # 聊天預設（配合 1260 字上限）
MAX_NEW_TOKENS_AUDIT = 640  # 監控診斷
MAX_NEW_TOKENS_VISUAL = 280 # 畫圖請求（文字短答）
MAX_PROMPT_TOKENS = 1280    # 縮短 prompt 可加快 prefill
MAX_OUTPUT_CHARS = 1260     # 硬性字數上限（含標點；不含 code fence 內 log 複製）
REPETITION_PENALTY = 1.15
NO_REPEAT_NGRAM = 4
# 設 LLM_4BIT=1 強制 4-bit（較省顯存、較慢）；預設 bf16 加速
FORCE_4BIT = os.environ.get("LLM_4BIT", "").strip().lower() in ("1", "true", "yes")

# 強制繁體中文輸出（所有 LLM 呼叫共用）
ZH_TW_OUTPUT_RULE = (
    "【語言硬性規定｜最高優先】你只能輸出「繁體中文」（zh-TW）。"
    "絕對禁止：日文、日本語、ひらがな、カタカナ、簡體中文、大段英文、全大寫英文牆。"
    "不可輸出日文結束語（如 答え終了），只能寫【回答結束】或【報告結束】。"
    "專有名詞（ISO 27001、RADIUS、SNMP、OOM、Linux）可保留原文，其餘說明必須繁中。"
    "【長度】全文 ≤ 1260 字，寫完即停，禁止同一份報告用兩種語言各寫一遍。"
    "【禁令】禁止字母亂碼、禁止提示詞洩漏、禁止括號內的寫作指示（如字數提醒）。"
    "【排版】只用 Markdown：## 標題、- 條列。"
)

OUTPUT_FORMAT_RULE = (
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
        "ctrl_id": "A.7.4 (Access)",
        "label": "A.7.4 驗證控制",
        "key": "access_control",
    },
    "TACACS": {
        "ctrl_id": "A.7.4 (Access)",
        "label": "A.7.4 驗證控制",
        "key": "access_control",
    },
}

CONTROL_TITLES = {
    "sec_gem_log": "A.8.24 密碼學與網絡傳輸安全校驗",
    "recipe_audit": "A.8.19 變更組態管理稽核",
    "access_control": "A.7.4 & A.11.2 實體安全與存取邊界事件",
    "patch_management": "A.8.8 技術弱點管理防禦日誌",
    "supplier_security": "A.5.19 供應商資安關係稽核預警",
    "malware_defense": "A.8.7 端點防範惡意軟體稽核",
}

# 減少碎片化；並開啟 TF32 加速 matmul（Ampere+ / Blackwell）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# =======================================================
# 1. 模型載入與初始化（預設 bf16 加速；可退回 4-bit）
# =======================================================
print("⏳ 正在載入工控/ISO27001 資安 LLM（速度模式）...")


def _model_is_prequantized(path: str) -> bool:
    cfg_path = Path(path) / "config.json"
    if not cfg_path.exists():
        return False
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        qc = cfg.get("quantization_config") or {}
        return bool(qc.get("load_in_4bit") or qc.get("_load_in_4bit"))
    except Exception:
        return False


def _load_llm():
    """
    載入策略（速度優先）:
    - 磁碟已是 4-bit merge → 直接載入，compute dtype 用 bf16
    - 否則優先 bf16 全精度（明顯快於 4-bit dequant）
    - LLM_4BIT=1 可強制 4-bit
    """
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    pre_q = _model_is_prequantized(MODEL_PATH)
    use_4bit = FORCE_4BIT or pre_q

    def _from_pretrained(attn_impl, **extra):
        kw = dict(device_map="auto", low_cpu_mem_usage=True, **extra)
        if attn_impl:
            kw["attn_implementation"] = attn_impl
        return AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kw)

    def _load_with_attn_fallback(**extra):
        try:
            return _from_pretrained("sdpa", **extra), "SDPA"
        except Exception as e1:
            print(f"⚠️ SDPA 載入失敗，改 eager：{e1}")
            return _from_pretrained(None, **extra), "eager"

    if use_4bit:
        extra = {}
        if not pre_q:
            extra["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        m, attn = _load_with_attn_fallback(**extra)
        mode = "4-bit（磁碟已量化）" if pre_q else "4-bit NF4"
        print(f"✅ LLM 載入成功（{mode} + {attn}）")
    else:
        try:
            m, attn = _load_with_attn_fallback(torch_dtype=compute_dtype)
            print(f"✅ LLM 載入成功（{compute_dtype} 全精度 + {attn}，最快）")
        except Exception as e:
            print(f"⚠️ 全精度載入失敗，改用 4-bit：{e}")
            m, attn = _load_with_attn_fallback(
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                )
            )
            print(f"✅ LLM 載入成功（4-bit NF4 後備 + {attn}）")

    m.eval()
    if hasattr(m, "config"):
        m.config.use_cache = True

    # 可選：首次編譯後解碼更快（設 LLM_COMPILE=1 開啟；首次請求會較久）
    if os.environ.get("LLM_COMPILE", "").strip().lower() in ("1", "true", "yes"):
        try:
            m = torch.compile(m, mode="reduce-overhead")
            print("⚡ 已啟用 torch.compile（首次推論會暖機）")
        except Exception as e:
            print(f"⚠️ torch.compile 略過：{e}")

    return m, tok


try:
    model, tokenizer = _load_llm()
except Exception as e:
    print(f"❌ 模型載入失敗，請確認 {MODEL_PATH} 路徑是否正確：{str(e)}")
    model, tokenizer = None, None

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
    text = reply.strip()
    if "【報告結束】" in text or "【回答結束】" in text:
        return False
    # 明顯寫到一半：結尾不是標點，或停在常見半截詞
    if text.endswith(("審", "建", "建置", "建置完", "建議", "防護", "不符", "摘要", "分析", "：", ":", "-", "、", "字段", "欄位", "以便", "加入")):
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


def _collapse_repetitions(text: str) -> str:
    """去除模型陷入的重複句 / 重複片段。"""
    if not text:
        return text

    # 1) 連續重複同一句（含「請您在 bar chart...」這類 loop）
    # 以句號/換行切段後去重保序
    parts = re.split(r'(?<=[。！？\n])', text)
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
    for window in (24, 32, 48, 64):
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
    ]
    for marker in loop_markers:
        first = text.find(marker)
        if first < 0:
            continue
        second = text.find(marker, first + len(marker))
        if second >= 0:
            text = text[:second].rstrip()

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


def _needs_zh_retry(text: str) -> bool:
    """輸出仍含日文、或中文過少時，觸發強制重寫。"""
    if not text or not text.strip():
        return True
    if _japanese_score(text) >= 3:
        return True
    if _has_japanese(text) and _cjk_count(text) < 40:
        return True
    # 幾乎沒有中文
    if _cjk_count(text) < 24 and len(text) > 40:
        return True
    return False


_PROMPT_LEAK_RE = re.compile(
    r"格式撰寫|撰寫診斷|撰寫報告|必要的資訊和建議|所有必要的資訊|"
    r"合作規診斷|請以[「\"'].*報告|回答第一行必須|總字數必須|"
    r"OUTPUT_FORMAT|禁止輸出「?##|只用繁體中文重寫|"
    r"請直接輸出繁體中文\s*Markdown|從「##\s*地端|"
    r"每段至少\s*1\s*句|不可只輸出\s*#|"
    r"【內部(?:監控|知識|日誌)|禁止照抄|轉述重點|"
    r"開始輸入答案|開始輸出答案|開始作答|輸入答案|"
    r"開始輸入|開始輸出|在此輸入|請開始寫|"
    r"一、二、三每段都要|禁止貼上日誌原文|勿虛構攻擊",
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
        """三段統一用段落＋精簡條列，避免樣式不一致。"""
        cleaned = _clean_lines(lines_list)
        if not cleaned:
            return fallback
        # 第一句當導讀段落，其餘（最多 3 條）當條列
        lead = cleaned[0]
        if len(lead) > 220:
            lead = lead[:220] + "…"
        rest = cleaned[1:4]
        if not rest:
            return lead
        bullets = "\n".join(
            f"- {(ln[:140] + '…') if len(ln) > 140 else ln}" for ln in rest
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
        s2.append("- 介面／供電狀態頻繁變動，反映可用性與實體連線不穩定，需納入營運風險追蹤。")
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

    return _build_report_cards(buckets["s1"], buckets["s2"], buckets["s3"])


def _looks_like_report(text: str) -> bool:
    """判斷是否為（或應整理成）三卡診斷報告。"""
    if not text:
        return False
    return bool(
        re.search(
            r"地端\s*LLM\s*智慧合規診斷報告|"
            r"##\s*一、\s*事件經過摘要|"
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

    replacements = [
        (r"(?m)^\s*#{0,4}\s*[*＊]*地端\s*LLM.*?報告[*＊]*\s*[:：]?\s*$", "## 地端 LLM 智慧合規診斷報告"),
        (r"(?m)^\s*#{0,4}\s*[*＊]*[一二三123]\s*[\.、．]?\s*事件經過摘要[*＊]*\s*$", "## 一、事件經過摘要"),
        (r"(?m)^\s*#{0,4}\s*[*＊]*[一二三123]\s*[\.、．]?\s*小報概述[*＊]*\s*$", "## 一、事件經過摘要"),
        (r"(?m)^\s*#{0,4}\s*[*＊]*事變總述[*＊]*\s*$", "## 一、事件經過摘要"),
        (r"(?m)^\s*[*＊]*事變總述[*＊]*\s*[:：]?\s*$", "## 一、事件經過摘要"),
        (r"(?m)^\s*#{0,4}\s*[*＊]*事件經過摘要[*＊]*\s*$", "## 一、事件經過摘要"),
        (r"(?m)^\s*#{0,4}\s*[*＊]*[一二三123]\s*[\.、．]?\s*.*(?:不合規|不符合標準|風險).*[*＊]*\s*$", "## 二、不合規／風險分析"),
        (r"(?m)^\s*#{0,4}\s*[*＊]*[一二三123]\s*[\.、．]?\s*.*不合規.*分析[*＊]*\s*$", "## 二、不合規／風險分析"),
        (r"(?m)^\s*#{0,4}\s*[*＊]*ISO\s*27001.*(?:分析|解析|評估)[*＊]*\s*$", "## 二、不合規／風險分析"),
        (r"(?m)^\s*#{0,4}\s*[*＊]*[一二三123]\s*[\.、．]?\s*.*(?:修補|防護|改善|修正|當事人).*建議[*＊]*\s*$", "## 三、具體修補建議"),
        (r"(?m)^\s*#{0,4}\s*[*＊]*具體修補與防護建議[*＊]*\s*$", "## 三、具體修補建議"),
    ]
    for pat, rep in replacements:
        text = re.sub(pat, rep, text, flags=re.I)

    text = re.sub(r"(可以看到如下[^\n：:]*[:：])\s*", r"\1\n", text)
    text = re.sub(r"(?<!\n)\s*([1-9][\)）])\s+", r"\n- ", text)
    text = re.sub(r"(?<!\n)\s*([1-9][\.、])\s+(?=[\u4e00-\u9fffA-Za-z])", r"\n- ", text)
    text = re.sub(r"([^\n])\s*(##\s+)", r"\1\n\n\2", text)
    text = re.sub(r"([。！？；])\s*[-•]\s+", r"\1\n- ", text)
    text = re.sub(r"([^\n])\s+(-\s+[\u4e00-\u9fffA-Za-z])", r"\1\n\2", text)
    text = re.sub(r"(##\s+[^\n]+)\n(?!\n|-)", r"\1\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 僅診斷報告才強制三卡；一般對話保持原樣
    if force_report or _looks_like_report(text):
        return _normalize_report_structure(text)
    return text


def _truncate_output(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """硬性字數上限；優先在句號處截斷。"""
    if not text or len(text) <= limit:
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


def normalize_llm_output(text: str, mode: str = "report") -> str:
    """統一後處理：去日文/亂碼 → 繁中 → 排版 → 1260 字（圖表不計入）。

    mode:
      - report: 診斷三卡（監控分析）
      - chat: 一般對話，不強制報告模板
    """
    if not text:
        return text
    mode = (mode or "report").lower()
    force_report = mode == "report"

    visual_blocks = re.findall(r"```(?:chart|mermaid)[\s\S]*?```", text, flags=re.I)
    body = re.sub(r"```(?:chart|mermaid)[\s\S]*?```", "", text, flags=re.I)

    body = _collapse_repetitions(body)
    body = _strip_code_fences(body)  # 先拿掉 code fence，避免拆成一堆卡
    body = _strip_context_leak(body)  # 先去掉照抄的 RAG／Log
    body = _strip_garbage(body)
    body = _to_traditional(body)
    body = _drop_japanese_lines(body)  # 轉繁後再清一次
    body = _strip_context_leak(body)  # 再清一次殘留前綴
    body = _beautify_markdown(body, force_report=force_report)
    body = _collapse_repetitions(body)
    body = _truncate_output(body, MAX_OUTPUT_CHARS)

    # 仍含日文或中文過少
    if _needs_zh_retry(body):
        zh_lines = [
            ln.strip() for ln in body.splitlines()
            if ln.strip() and not _has_japanese(ln) and _cjk_count(ln) >= 6
            and not ln.strip().startswith("#")
        ]
        if force_report:
            body = _fallback_zh_report(" ".join(zh_lines[:3]))
        elif zh_lines:
            body = "\n".join(zh_lines[:6])
        else:
            body = "抱歉，剛才輸出異常。請再問一次，或改問合規現況／日誌分析。"

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
    return body.strip()


def _finish_incomplete_sentence(text: str) -> str:
    """若結尾明顯半截，補上收束句，避免畫面停在半個英文單字。"""
    if not text:
        return text
    t = text.rstrip()
    if t.endswith(("。", "！", "？", "】", "`")):
        return t
    # 去掉尾端半截英數碎片
    t = re.sub(r'[\sA-Za-z0-9_\-\'\"\.]+$', '', t).rstrip("，,、:：")
    if not t.endswith(("。", "！", "？")):
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


def run_llm(messages, max_new_tokens=MAX_NEW_TOKENS, allow_continue=False, output_mode="report"):
    """通用 LLM 推論：接收 chat messages，回傳模型文字回應。"""
    if model is None or tokenizer is None:
        return "模型未成功載入，無法提供 AI 診斷。"
    output_mode = (output_mode or "report").lower()

    try:
        def _generate_once(msgs, token_budget):
            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                parts = []
                for msg in msgs:
                    parts.append(f"<|{msg['role']}|>\n{msg['content']}<|end|>\n")
                parts.append("<|assistant|>\n")
                prompt = "".join(parts)

            local_inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_PROMPT_TOKENS
            )
            device = next(model.parameters()).device
            local_inputs = {k: v.to(device) for k, v in local_inputs.items()}

            input_token_len = local_inputs["input_ids"].shape[1]
            print(f"🧮 prompt_tokens={input_token_len}, max_new_tokens={token_budget}")

            stopping = StoppingCriteriaList([
                _StopOnStrings(_stop_ids_list, input_token_len)
            ])

            with torch.inference_mode():
                local_outputs = model.generate(
                    **local_inputs,
                    max_new_tokens=token_budget,
                    do_sample=False,
                    repetition_penalty=REPETITION_PENALTY,
                    no_repeat_ngram_size=NO_REPEAT_NGRAM,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    stopping_criteria=stopping,
                )

            new_tokens = local_outputs[0][input_token_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            return _collapse_repetitions(text)

        reply = _generate_once(messages, max_new_tokens)
        reply = _collapse_repetitions(reply)

        # 續寫最多 1 次（避免 2～3 倍延遲）；僅明顯截斷時觸發
        if allow_continue and _looks_truncated(reply):
            print("⚠️ 偵測到回答可能被截斷，自動續寫補完（1 次）...")
            clipped = _collapse_repetitions(reply)
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
            continuation = _generate_once(
                continue_messages,
                min(220, max_new_tokens // 2 + 64),
            )
            continuation = _collapse_repetitions(continuation)
            if continuation:
                if continuation.startswith(clipped[:24]):
                    reply = continuation
                else:
                    reply = (clipped.rstrip() + "\n" + continuation.lstrip()).strip()
                reply = _collapse_repetitions(reply)

        if _looks_truncated(reply):
            reply = _finish_incomplete_sentence(reply)

        cleaned = normalize_llm_output(reply, mode=output_mode)

        # 淨化後仍不合格（日文殘留／落到攔截模板）才強制繁中重寫一次
        if _needs_zh_retry(cleaned) or _has_japanese(cleaned) or "已自動攔截" in cleaned:
            print("⚠️ 偵測到非繁中／日文混雜，強制以繁體中文重寫一次...")
            if output_mode == "chat":
                rewrite_hint = (
                    "【強制重寫】上一則無效。請只用繁體中文、像一般對話簡短重答，"
                    "禁止日文，禁止輸出「智慧合規診斷報告」三卡格式；≤400字；最後【回答結束】。"
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
            retry_raw = _generate_once(retry_messages, min(max_new_tokens, 480))
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
        print("⚠️ 發生 CUDA Out of Memory！自動強制清理顯存與垃圾回收...")
        torch.cuda.empty_cache()
        gc.collect()
        return "分析失敗：Log 數據量超出 GPU 顯存負荷，已自動清理記憶體，請削減 Log 長度後重試。"

    except Exception as e:
        print(f"LLM 推理發生錯誤: {str(e)}")
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
        r"智慧合規診斷報告.{0,12}開始",
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


def ask_llm(log_text, control_key=None, title=None, metric_summary=None, rag_context=None):
    """
    呼叫 LLM 對傳入的 Log / 控制項現況進行智能分析與合規剖析
    """
    control_title = title or CONTROL_TITLES.get(control_key or "", "ISO 27001 控制項")

    if is_empty_log(log_text) and not metric_summary:
        return build_factual_report(
            control_title=control_title,
            metric_summary="目前無量化摘要",
            log_text="",
            rag_context=rag_context,
        )

    # 1. 安全截斷日誌
    log_text = log_text or ""
    if len(log_text) > MAX_INPUT_CHARS:
        log_text = log_text[:MAX_INPUT_CHARS] + "\n...[Log 過長已截斷]..."

    # 有真實 TXT 事件時，先備好事實三卡；模型若再洩漏提示詞就直接採用
    factual_ready = None
    if _has_real_log_signal(log_text) or (metric_summary and "count=" in str(metric_summary)):
        factual_ready = build_factual_report(
            control_title=control_title,
            metric_summary=metric_summary,
            log_text=log_text,
            rag_context=rag_context,
        )

    metric_part = f"\n控制項量化摘要：{metric_summary}" if metric_summary else ""
    rag_part = (
        f"\n【內部知識要點｜禁止照抄】請轉述下列重點，勿貼原文：\n{rag_context}\n"
        if rag_context else ""
    )

    # 2. 構建結構化 Prompt（避免「請從某某開始輸出」被模型照抄）
    messages = [
        {
            "role": "system",
            "content": (
                f"{ZH_TW_OUTPUT_RULE}"
                "你是 OT/ISO 27001 資安稽核專家。只寫診斷結論，不要複誦任何指示句。"
                f"{OUTPUT_FORMAT_RULE}"
                "每段必須寫「實際觀察到的風險／事件／建議」，禁止寫「開始輸入／開始輸出／格式撰寫」。"
                "總字數 ≤ 1260；日誌只能轉述重點，禁止整段複製；不要輸出「## AI:」。"
                "無事件時依量化摘要評估，勿虛構攻擊。"
            )
        },
        {
            "role": "user",
            "content": (
                f"控制項：{control_title}\n"
                f"{metric_part}\n"
                f"{rag_part}"
                f"日誌重點（勿照抄原文，改寫成繁中結論）：\n{log_text or '（無事件明細）'}\n"
                "請產出三個段落："
                "（1）事件經過摘要（2）不合規／風險分析（3）具體修補建議。"
                "每段至少一句具體內容。最後一行【報告結束】。"
            )
        }
    ]
    reply = run_llm(messages, max_new_tokens=MAX_NEW_TOKENS_AUDIT, allow_continue=False)

    # 模型輸出過差／提示詞洩漏／無領域內容 → 改用 TXT 事實三卡
    if _report_is_weak(reply) or _report_lacks_domain_content(reply):
        print("⚠️ LLM 報告內容過弱或提示詞洩漏，改以 TXT／指標事實模板補齊三卡")
        factual = factual_ready or build_factual_report(
            control_title=control_title,
            metric_summary=metric_summary,
            log_text=log_text,
            rag_context=rag_context,
        )
        return factual
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
        "介紹一下自己", "自我介紹",
    }
    if compact in casual:
        return True
    if len(t) <= 16 and re.fullmatch(
        r"(你好|您好|哈囉|嗨|hello|hi|hey)(嗎|呀|啊|嘛)?[!！?？.。]*",
        t,
        re.I,
    ):
        return True
    return False


CASUAL_CHAT_REPLY = (
    "你好！我是 Semi-Shield Cyber Agent，專注 OT 工控資安與 ISO 27001 合規協助。\n\n"
    "你可以這樣問我：\n"
    "- 目前合規現況如何？\n"
    "- 幫我分析 SYSLOG／RADIUS 風險\n"
    "- 畫出合規狀態圖表\n\n"
    "直接輸入需求即可。【回答結束】"
)


def wants_report_format(user_message: str) -> bool:
    """是否應輸出三卡診斷報告（監控／合規分析）。"""
    if is_casual_chat(user_message):
        return False
    if wants_visual(user_message):
        return False
    keywords = [
        "診斷", "報告", "缺失", "不符", "合規", "稽核", "日誌", "log", "威脅", "入侵",
        "隨身碟", "usb", "機台", "組態", "radius", "tacacs", "snmp", "syslog",
        "存取", "驗證", "malware", "惡意", "現況", "狀態", "控制項", "iso",
        "ot", "掃描", "事件", "breach", "patch", "修補", "風險", "不合規",
        "分析一下", "幫我看", "檢查",
    ]
    text = (user_message or "").lower()
    return any(k in text for k in keywords)


def needs_ot_context(user_message):
    """依關鍵字判斷是否需要下探 OT 日誌資料。"""
    if is_casual_chat(user_message):
        return False
    keywords = [
        "nc", "缺失", "不符", "合規", "稽核", "日誌", "log", "威脅", "入侵",
        "隨身碟", "usb", "機台", "組態", "radius", "tacacs", "snmp", "syslog",
        "存取", "驗證", "malware", "惡意", "現況", "狀態", "控制項", "iso",
        "ot", "掃描", "資料庫", "事件", "breach", "patch", "修補",
        "圖表", "圖形", "視覺化", "趨勢", "統計", "chart", "pie", "bar",
        "流程圖", "架構圖", "mermaid"
    ]
    text = user_message.lower()
    return any(k in text for k in keywords)


STATUS_SCORE = {"pass": 0, "review": 1, "fail": 2, "compliant": 0, "attention": 1}

CONTROL_LABEL_ALIAS = {
    "sec_gem_log": "A.8.24",
    "recipe_audit": "A.8.19",
    "access_control": "A.7.4",
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
        (("流程", "架構", "步驟", "mermaid", "flowchart"), "mermaid"),
    ]
    for keys, chart_type in mapping:
        if any(k in text for k in keys):
            types.append(chart_type)

    # 「各種 / 多種 / 圖形」→ 一次產出多種類型
    want_many = any(k in text for k in (
        "各種", "多種", "多個圖", "不同圖", "圖形", "視覺化", "圖表"
    ))
    if want_many and not types:
        types = ["bar", "pie", "line", "radar", "doughnut", "mermaid"]
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
    data = scan_ot_directory()
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
    """流程/架構圖後備。"""
    if not force:
        flow_keys = ["流程", "架構", "步驟", "flowchart", "mermaid", "怎麼修", "如何修", "各種", "多種", "圖形"]
        text = (user_message or "").lower()
        if not any(k in text for k in flow_keys):
            return None
    return (
        "\n\n```mermaid\n"
        "flowchart TD\n"
        "A[蒐集 OT 日誌] --> B[對照 ISO 27001 控制項]\n"
        "B --> C{是否存在 NC / 風險}\n"
        "C -->|是| D[產出修補建議]\n"
        "C -->|否| E[維持監控]\n"
        "D --> F[驗證與複核]\n"
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
    data = scan_ot_directory()
    if data is None:
        return "OT 目錄不存在，尚無可用日誌。"
    if isinstance(data, dict) and "error" in data:
        return data["error"]

    metrics = data.get("metrics", {})
    logs = data.get("parsed_logs", [])[:8]
    lines = ["【目前 OT / ISO 27001 監控摘要】"]
    for key, meta in metrics.items():
        lines.append(f"- {key}: {meta.get('text')} (status={meta.get('status')})")

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


def ask_agent(user_message, ot_context=None, rag_context=None):
    """聊天 Agent：可選擇附帶 OT 掃描結果與 RAG 知識庫上下文。"""
    visual = wants_visual(user_message)
    report_mode = wants_report_format(user_message) and not visual
    output_mode = "report" if report_mode else "chat"

    if visual:
        # 畫圖請求：要求短而完整的文字，圖表由後端 ensure_visual_reply 補強
        system_content = (
            f"{ZH_TW_OUTPUT_RULE}"
            "你是 Semi-Shield Cyber Agent。使用者要看圖表。"
            "請只用繁體中文寫 3-6 句完整說明（合規現況與重點風險），總字數 ≤ 1260。"
            "句子必須寫完，最後一行【回答結束】。"
            "不要輸出 Chart.js 教學、不要重複句子、不要寫半截英文或字母亂碼。"
            "可選附上一個簡短 ```chart```；若無法穩定產出也可只寫文字。"
            "禁止輸出「智慧合規診斷報告」三卡格式。"
        )
    elif report_mode:
        system_content = (
            f"{ZH_TW_OUTPUT_RULE}"
            "你是 Semi-Shield Cyber Agent，專精於 OT 工控資安與 ISO 27001 合規稽核。"
            "請以繁體中文簡潔、專業地回答，並使用 Markdown 標題與條列美化排版。"
            f"{OUTPUT_FORMAT_RULE}"
            "總字數必須 ≤ 1260。若提供監控摘要或 RAG，請轉述重點；勿虛構。"
            "知識不足時請明說。"
            f"{CHART_FORMAT_HINT}"
            "【絕對禁令】：禁止複製／貼上 Log、RAG 原文、監控流水帳；"
            "禁止輸出「## AI:」；禁止字母亂碼、大段英文牆、簡體中文。"
            "回答第一行必須是「## 地端 LLM 智慧合規診斷報告」。"
        )
    else:
        system_content = (
            f"{ZH_TW_OUTPUT_RULE}"
            "你是 Semi-Shield Cyber Agent，專精 OT 工控資安與 ISO 27001。"
            "這是一般對話／知識問答：用繁體中文自然回覆，2-8 句即可。"
            "可用簡短條列，但【禁止】輸出「地端 LLM 智慧合規診斷報告」或一／二／三卡格式。"
            "勿虛構監控事件；知識不足請明說。總字數 ≤ 800。最後一行【回答結束】。"
        )

    parts = []
    if ot_context:
        # 壓縮監控摘要，降低模型照抄原文
        ctx = ot_context
        limit = 480 if visual else 720
        if len(ctx) > limit:
            ctx = ctx[:limit] + "\n...(摘要已截斷)..."
        parts.append(
            "【內部監控摘要｜禁止照抄原文，只能轉述重點】\n"
            f"{ctx}"
        )
    if rag_context and not visual:
        parts.append(
            "【內部知識要點｜禁止照抄，只能用自己的話寫繁中結論】\n"
            f"{rag_context}"
        )
    parts.append(f"使用者問題：{user_message}")
    if visual:
        parts.append(
            "請用繁體中文完整說明合規狀態重點（3-6 句），不要重複，最後寫【回答結束】。"
            "圖表會由系統自動補上。不要貼日誌或 RAG 原文。"
        )
    elif report_mode:
        parts.append(
            "請直接輸出繁體中文 Markdown 報告（從「## 地端 LLM 智慧合規診斷報告」開始），"
            "總字數 ≤ 1260；不要貼日誌／RAG／監控原文；最後寫【報告結束】。"
        )
    else:
        parts.append(
            "請以一般對話方式用繁體中文簡答；不要輸出診斷報告三卡；最後寫【回答結束】。"
        )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
    # 畫圖請求：文字短答即可；一般聊天也不自動續寫（避免雙倍延遲）
    token_budget = MAX_NEW_TOKENS_VISUAL if visual else (
        MAX_NEW_TOKENS_AUDIT if report_mode else min(MAX_NEW_TOKENS, 320)
    )
    return run_llm(
        messages,
        max_new_tokens=token_budget,
        allow_continue=False,
        output_mode=output_mode,
    )


def retrieve_rag_for_query(query: str):
    """統一 RAG 檢索，回傳 (context_text, citations, hits)。失敗時回空，不中斷對話。"""
    try:
        hits = rag_service.retrieve(query) or []
        context = rag_service.format_context(hits)
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
        r"LOCAL_LOGIN|LOGIN_FAILED|LOGIN_SUCCESS|TTY_EXPIRE",
    ),
    # 傳輸／加密／SNMP
    (
        "sec_gem_log",
        r"SNMP|CRYPTO|PKI|TLS|SSL|IPSEC|SSH_SESSION|CERTIFICATE|SUDI",
    ),
    # 弱點／韌體／映像
    (
        "patch_management",
        r"BOOT|IMAGE|VERSION|PLATFORM|SOFTWARE|UPGRADE|IOSXE|INSTALL",
    ),
    # 供應鏈／外部連線／遠端 logging
    (
        "supplier_security",
        r"CDP|LLDP|NEIGHBOR|LOGGING\s+TO|TRAP|REMOTE.?HOST|PNP_",
    ),
    # 惡意／異常防護（若日誌有）
    (
        "malware_defense",
        r"IPS|IDS|MALWARE|VIRUS|THREAT|PORTSCAN|DOS|DDOS",
    ),
    # 其餘組態／介面／PoE → 組態變更稽核（A.8.19）
    (
        "recipe_audit",
        r"CONFIG|SYS-|LINK-|LINEPROTO|ILPOWER|PARSER|DUAL_ACTIVE|"
        r"SPANTREE|PORT_SECURITY|ERR_DISABLE|UPDOWN|POWER_",
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
        "access_control": ("A.7.4 (Access)", "A.7.4 驗證控制"),
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
    """
    device, ip = _parse_device_from_txt_name(path.name)
    counts_by_key = {k: 0 for k in CONTROL_TITLES}
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

                recent.append(ev)
                if len(recent) > max_events_per_file * 2:
                    recent = recent[-max_events_per_file:]

                is_prio = (
                    key in ("access_control", "sec_gem_log", "malware_defense")
                    or ev.get("severity", 5) <= 3
                    or re.search(
                        r"CONFIG_I|LOGIN_FAILED|DENIED|RADIUS|SNMP-3|MALWARE|PSECURE",
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
    }
    if event_lines > len(samples):
        meta["truncated_to"] = len(samples)
    return samples, meta


def build_control_log_bundle(parsed_logs_list, metrics, max_lines=6):
    """為各控制項組裝給 LLM 的精簡日誌摘要（控制長度以免擠掉完整回答）。"""
    bundles = {}
    for key in CONTROL_TITLES:
        matched = [x for x in parsed_logs_list if x.get("key") == key]
        metric = metrics.get(key, {})
        full_count = int(metric.get("count", len(matched)) or 0)
        metric_summary = (
            f"count={full_count}, "
            f"status={metric.get('status', 'n/a')}, "
            f"text={metric.get('text', 'n/a')}"
        )
        if matched:
            # 優先較嚴重／較新
            ranked = sorted(
                matched,
                key=lambda x: (x.get("severity", 5), x.get("time", "")),
            )
            lines = [m["raw"] for m in ranked[:max_lines]]
            log_text = "\n".join(lines)
            omitted = max(0, full_count - len(lines))
            if omitted > 0:
                log_text += f"\n...另有 {omitted} 筆同類事件已省略..."
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
            events, meta = _parse_txt_log_file(path)
            file_metas.append(meta)
            parsed_logs_list.extend(events)
            # 全量計數（非僅樣本）
            for k, n in (meta.get("counts_by_key") or {}).items():
                if k in metrics:
                    metrics[k]["count"] += int(n or 0)
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

    control_bundles = build_control_log_bundle(display_logs, metrics)

    return {
        "metrics": metrics,
        "parsed_logs": display_logs,
        "all_logs_content": all_logs_content,
        "control_bundles": control_bundles,
    }

# =======================================================
# 3. 前端頁面與 API 路由設定
# =======================================================
@app.route('/')
@app.route('/platform')
def serve_platform_page():
    """整合平台首頁：監控戰情 + AI 對話。"""
    return send_from_directory(BASE_DIR, 'platform.html')


@app.route('/chat')
def serve_chat_page():
    """提供前端聊天介面網頁（可被整合平台嵌入）。"""
    return send_from_directory(BASE_DIR, 'agent_chat.html')


@app.route('/monitor')
def serve_monitor_page():
    """提供原有稽核監控儀表板（可被整合平台嵌入）。"""
    return send_from_directory(BASE_DIR, 'OT.html')


@app.route('/api/monitor/data', methods=['GET'])
def get_monitor_data():
    data = scan_ot_directory()
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

    # 若前端只丟 control_key，或 log 為空，自動補齊最新 bundle
    if control_key or is_empty_log(log_content):
        data = scan_ot_directory()
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

    if model is None or tokenizer is None:
        return jsonify({
            "error": "模型未成功載入，無法提供 AI 診斷。請檢查後端啟動日誌。",
            "status": "error"
        }), 503

    # 監控戰情診斷略過護欄（護欄僅套用於 /api/agent/chat 對話）
    rag_query = f"{title or control_key or ''} ISO 27001 {str(log_content or '')[:240]}"
    rag_context, rag_citations, _ = retrieve_rag_for_query(rag_query)

    print(f"🔍 單項診斷: key={control_key} title={title} rag_hits={len(rag_citations)}")
    ai_analysis = ask_llm(
        log_content,
        control_key=control_key,
        title=title,
        metric_summary=metric_summary,
        rag_context=rag_context or None,
    )

    return jsonify({
        "status": "success",
        "control_key": control_key,
        "raw_log": log_content,
        "ai_analysis": ai_analysis,
        "guardrail": {"skipped": True, "reason": "監控戰情略過護欄"},
        "rag_enabled": rag_service.enabled,
        "rag_citations": rag_citations,
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

    data = scan_ot_directory()
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

    if model is None or tokenizer is None:
        return jsonify({
            "error": "模型未成功載入，無法提供 AI 診斷。請檢查後端啟動日誌。",
            "status": "error"
        }), 503

    results = {}
    rag_meta = {}
    for key, job in jobs.items():
        print(f"正在分析控制項: {key} ...")
        # 監控戰情批次診斷略過護欄（護欄僅套用於對話）
        rag_query = f"{job.get('title') or key} ISO 27001 {str(job.get('log') or '')[:240]}"
        rag_context, rag_citations, _ = retrieve_rag_for_query(rag_query)
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
        "rag_enabled": rag_service.enabled,
        "meta": rag_meta,
    })


@app.route('/api/agent/chat', methods=['POST'])
def agent_chat():
    """前端聊天介面：護欄 → RAG 檢索 → LLM → 圖表補強 → 輸出脫敏。"""
    req_data = request.get_json(silent=True) or {}
    user_message = (req_data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "請傳入 JSON：{'message': '你的問題'}"}), 400

    # 1) 輸入護欄
    guard = guardrail_service.check_input(user_message)
    if guard.get("blocked"):
        print(f"🛡️ 護欄攔截: {guard.get('reason')} | {user_message[:80]}")
        return jsonify({
            "status": "blocked",
            "reply": guardrail_service.block_message(guard),
            "tool_called": False,
            "tool_name": "guardrail",
            "tool_query": guard.get("mode"),
            "guardrail": guard,
            "rag_enabled": rag_service.enabled,
            "rag_citations": [],
        }), 200

    # 寒暄：不跑 RAG／不套診斷報告，直接簡短回覆
    if is_casual_chat(user_message):
        print(f"💬 寒暄模式（略過 RAG／報告）| {user_message[:40]}")
        reply, redacted = guardrail_service.sanitize_output(CASUAL_CHAT_REPLY)
        return jsonify({
            "status": "success",
            "reply": reply,
            "tool_called": False,
            "tool_name": None,
            "tool_query": None,
            "guardrail": {
                "blocked": False,
                "mode": guard.get("mode") or guardrail_service.mode,
                "output_redacted": redacted,
                "safe_prob": guard.get("safe_prob", 1.0),
                "unsafe_prob": guard.get("unsafe_prob", 0.0),
                "reason": guard.get("reason"),
            },
            "rag_enabled": rag_service.enabled,
            "rag_citations": [],
            "chat_mode": "casual",
        })

    tool_called = False
    tool_name = None
    tool_query = None
    ot_context = None
    report_mode = wants_report_format(user_message)

    if needs_ot_context(user_message):
        tool_called = True
        tool_name = "scan_ot_directory"
        tool_query = "metrics + recent_parsed_logs"
        print(f"🤖 Agent Tool Call: {tool_name} | 問題: {user_message[:80]}")
        ot_context = build_ot_context_summary()

    # RAG：診斷／技術問答才檢索；避免「你好」撈到無關事件
    rag_context, rag_citations = "", []
    if report_mode or needs_ot_context(user_message) or len(user_message) >= 8:
        # 短句知識問（如「什麼是 RADIUS」）仍可 RAG；純寒暄已在上方攔截
        use_rag = report_mode or bool(
            re.search(
                r"什麼|如何|怎麼|為何|定義|說明|介紹|差異|iso|radius|snmp|syslog|ot|合規|控制",
                user_message,
                re.I,
            )
        )
        if use_rag:
            rag_context, rag_citations, _ = retrieve_rag_for_query(user_message)
            if rag_citations:
                tool_called = True
                if tool_name:
                    tool_name = f"{tool_name}+rag_retrieve"
                    tool_query = f"{tool_query}; top_k={len(rag_citations)}"
                else:
                    tool_name = "rag_retrieve"
                    tool_query = f"top_k={len(rag_citations)}"
                print(f"📚 RAG hits={len(rag_citations)} | {user_message[:80]}")

    # 3) LLM 生成
    reply = ask_agent(
        user_message,
        ot_context=ot_context,
        rag_context=rag_context or None,
    )
    reply = ensure_visual_reply(user_message, reply)
    # 圖表注入後再淨化一次（文字 ≤1260、繁中、去亂碼；chart 區塊保留）
    reply = normalize_llm_output(reply, mode="report" if report_mode else "chat")

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
        "rag_enabled": rag_service.enabled,
        "rag_citations": rag_citations,
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


def _discover_local_llm_models() -> list[dict]:
    """掃描 train_ai/models、train_ai/train_llm 與舊路徑。"""
    root = Path(BASE_DIR)
    train_ai = root / "train_ai"
    train_llm = train_ai / "train_llm"
    models_dir = train_ai / "models"
    items: list[dict] = []
    seen: set[str] = set()

    def add(slug: str, path: Path):
        key = str(path.resolve())
        if key in seen or not (path / "config.json").is_file():
            return
        seen.add(key)
        meta = {}
        meta_path = path / "train_meta.json"
        if meta_path.is_file():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f) or {}
            except Exception:
                meta = {}
        items.append({
            "slug": slug,
            "path": str(path.resolve()),
            "base_model_id": meta.get("base_model_id"),
            "active": str(path.resolve()) == str(Path(MODEL_PATH).resolve()),
        })

    for base in (models_dir, train_llm):
        if not base.is_dir():
            continue
        for p in sorted(base.iterdir()):
            if p.is_dir() and (p / "config.json").is_file():
                add(p.name, p)

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
                add(slug, c)
                break
    return items


@app.route('/api/llm/models', methods=['GET'])
def list_llm_models():
    """列出可切換的本地微調模型；實際載入需設 LLM_MODEL_PATH 後重啟。"""
    models = _discover_local_llm_models()
    return jsonify({
        "current": MODEL_PATH,
        "models": models,
        "hint": "切換：set LLM_MODEL_PATH=<path> 後重啟 python app.py；比較：cd train_ai/train_llm && python compare_models.py --models a,b",
    })


@app.route('/api/safety/status', methods=['GET'])
def safety_status():
    """回傳 RAG / 護欄 / 離線資源就緒狀態，方便前端顯示。"""
    offline = _offline_vendor_status()
    return jsonify({
        "rag_enabled": rag_service.enabled,
        "rag_docs": len(getattr(rag_service, "docs", []) or []),
        "guardrail_mode": "disabled" if not ENABLE_GUARDRAIL else guardrail_service.mode,
        "guardrail_enabled": ENABLE_GUARDRAIL,
        "guardrail_model_dir": str(
            (Path(BASE_DIR) / "train_ai" / "train_gur" / "fine_tuned_guardrail")
        ),
        "offline_ready": offline["ready"],
        "offline": offline,
        "llm_model_path": MODEL_PATH,
        "llm_models": _discover_local_llm_models(),
    })


# =======================================================
# 4. 伺服器啟動入口
# =======================================================
if __name__ == '__main__':
    if not os.path.exists(OT_FOLDER):
        os.makedirs(OT_FOLDER)
    offline = _offline_vendor_status()
    print("\n🚀 Semi-Shield 智慧網路資安與 LLM 整合後端伺服器已啟動。")
    print("🏠 整合平台：http://127.0.0.1:2000/  或  http://127.0.0.1:2000/platform")
    print("💬 聊天介面：http://127.0.0.1:2000/chat")
    print("📊 監控儀表板：http://127.0.0.1:2000/monitor")
    print(f"📚 RAG：{'啟用' if rag_service.enabled else '停用'}")
    print(f"🛡️ 護欄：{'啟用 (' + guardrail_service.mode + ')' if ENABLE_GUARDRAIL else '停用'}")
    print(
        f"📴 離線前端套件：{'就緒' if offline['ready'] else '缺少 ' + ','.join(offline['missing'])}"
    )
    app.run(host='0.0.0.0', port=2000, debug=False)