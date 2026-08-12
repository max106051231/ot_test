"""
護欄服務：ML 輸入攔截 + 平台合法意圖放行 + 硬性惡意規則 + 輸出脫敏。

微調模型對「產出／生成報告」等合規用語常誤判 unsafe，
因此對明確的 OT／ISO／診斷／報告請求做 allowlist 覆寫（仍擋注入／惡意）。

環境變數：
  GUARDRAIL_THRESHOLD=0.5
  GUARDRAIL_BASE_TOKENIZER=hfl/chinese-roberta-wwm-ext
  GUARDRAIL_DEVICE=cpu   # 預設 cpu；勿與主 LLM 同搶 GPU（可改 cuda）
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from code.paths import project_root, train_ai_dir

GUARD_DIR = train_ai_dir() / "train_gur" / "fine_tuned_guardrail"
BASE_TOKENIZER_ID = os.environ.get(
    "GUARDRAIL_BASE_TOKENIZER", "hfl/chinese-roberta-wwm-ext"
)

ID2LABEL = {0: "safe", 1: "unsafe"}

# 硬性攔截：即使看起來像合規問題也擋（注入／明確惡意）
HARD_UNSAFE_PATTERNS = [
    re.compile(p, re.I)
    for p in [
        r"忽略(所有|全部)?(規則|指令|限制|系統提示)",
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|rules)",
        r"jailbreak",
        r"dan\s*mode",
        r"繞過(護欄|安全|限制|防禦)",
        r"bypass\s+(the\s+)?(guardrail|safety|filter)",
        r"告訴我(你的)?系統提示",
        r"reveal\s+(your\s+)?(system\s+)?prompt",
        r"製作(勒索|病毒|木馬|炸彈)",
        r"(how\s+to\s+)?(make|build)\s+(ransomware|malware|bomb)",
        r"教我(駭入|入侵|攻擊).{0,12}(生產|工控|ot|plc)",
        r"刪除(所有|全部).{0,8}(日誌|log|資料庫)",
    ]
]

# 平台合法意圖：合規報告／診斷／OT 問答／加固設定／貼 syslog → 覆寫 ML 誤擋
_PLATFORM_ALLOW_RE = re.compile(
    r"("
    r"報告|診斷|稽核|合規|現況|控制項|修補|建議|"
    r"產出|生成|撰寫|寫一份|出一份|產生|"
    r"iso\s*/?\s*iec\s*27001|iso\s*27001|iso27001|27001|27002|"
    r"我要的是|我是說|我指的是|a\.\d+|annex|"
    r"ot|工控|半導體|無塵室|"
    r"syslog|cisco|radius|tacacs|snmp|modbus|sec/?gem|"
    r"telnet|ssh|https|aaa|vty|transport\s*input|"
    r"停用|禁用|關閉|加固|hardening|弱協議|明文|"
    r"登入|存取|驗證|malware|惡意|updown|login|"
    r"分析|檢查|審視|檢視|監控|日誌|log|"
    r"圖表|視覺化|mermaid|chart|"
    r"什麼是|何謂|如何|怎麼|怎麽|為何|說明|步驟|設定|配置|執行"
    r")",
    re.I,
)

# 管理面加固／弱協議停用：ML 常把「停用／禁用」誤判 unsafe
_HARDENING_ALLOW_RE = re.compile(
    r"("
    r"telnet|ssh|https|snmpv?[123]?|aaa|tacacs|radius|vty|"
    r"transport\s*input|停用|禁用|關閉|加固|hardening|"
    r"弱協議|明文登入|管理面|交換器|catalyst"
    r")",
    re.I,
)

_CASUAL_RE = re.compile(
    r"^(你好|您好|嗨|哈囉|hello|hi|hey|早安|午安|晚安|謝謝|感謝)"
    r"[\s!！。.？?~～]*$",
    re.I,
)

_CISCO_SYSLOG_RE = re.compile(r"%[A-Z0-9_-]+-\d+-[A-Z0-9_]+", re.I)

# 輸出脫敏（非輸入規則判斷）
REDACT_PATTERNS = [
    (re.compile(r"\bwlc-\d{4}-\d{2}\b", re.I), "wlc-****-**"),
    (re.compile(r"\b(?:password|passwd|pwd)\s*[:=]\s*\S+", re.I), "password=[REDACTED]"),
    (re.compile(r"\b(?:api[_-]?key|secret|token)\s*[:=]\s*\S+", re.I), "secret=[REDACTED]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "*.*.*.*"),
]


class GuardrailService:
    def __init__(self, threshold: float | None = None):
        env_th = os.environ.get("GUARDRAIL_THRESHOLD")
        if threshold is not None:
            self.threshold = float(threshold)
        elif env_th:
            self.threshold = float(env_th)
        else:
            self.threshold = 0.5
        self.mode = "disabled"
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self._init_model()

    def _tokenizer_is_usable(self, tokenizer) -> bool:
        """缺 vocab 時會把中文全編成 [UNK]，導致判斷失真。"""
        try:
            if len(tokenizer) < 1000:
                return False
            sample = "什麼是 ISO 27001 與 OT 資安"
            ids = tokenizer.encode(sample, add_special_tokens=True)
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if unk_id is None or not ids:
                return True
            skip = {
                getattr(tokenizer, "cls_token_id", None),
                getattr(tokenizer, "sep_token_id", None),
                getattr(tokenizer, "pad_token_id", None),
            }
            content = [i for i in ids if i not in skip]
            if not content:
                return False
            unk_ratio = sum(1 for i in content if i == unk_id) / len(content)
            return unk_ratio < 0.5
        except Exception:
            return False

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        try:
            tok = AutoTokenizer.from_pretrained(str(GUARD_DIR))
            if self._tokenizer_is_usable(tok):
                return tok, "local"
            print(
                "⚠️ 護欄：本地 tokenizer 異常，改用基底 "
                f"{BASE_TOKENIZER_ID}"
            )
        except Exception as e:
            print(f"⚠️ 護欄：讀取本地 tokenizer 失敗：{e}")

        tok = AutoTokenizer.from_pretrained(BASE_TOKENIZER_ID)
        if not self._tokenizer_is_usable(tok):
            raise RuntimeError("基底 tokenizer 不可用")
        try:
            tok.save_pretrained(str(GUARD_DIR))
            print(f"✅ 護欄：已將 tokenizer 補存至 {GUARD_DIR}")
        except Exception as e:
            print(f"⚠️ 護欄：無法寫回 tokenizer 檔：{e}")
        return tok, "base"

    def _init_model(self):
        # Edge／明確關閉：不載入 ML，節省樹莓派 RAM
        if os.environ.get("ENABLE_GUARDRAIL", "1").strip().lower() in (
            "0", "false", "no", "off",
        ):
            print("🛡️ 護欄：已停用（ENABLE_GUARDRAIL=0），輸入檢查放行")
            self.mode = "disabled"
            return
        if not GUARD_DIR.exists() or not (GUARD_DIR / "config.json").is_file():
            print(f"⚠️ 護欄：找不到微調模型 {GUARD_DIR}，改為規則模式（硬性規則 + 平台放行）。")
            self.mode = "rules"
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification

            self.tokenizer, tok_src = self._load_tokenizer()
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(GUARD_DIR)
            )
            self.model.eval()
            # 預設 CPU：避免與主 LLM 搶同一張 GPU 造成 CUDA illegal memory access
            # 若要護欄上 GPU：設 GUARDRAIL_DEVICE=cuda
            pref = (os.environ.get("GUARDRAIL_DEVICE") or "cpu").strip().lower()
            if pref in ("cuda", "gpu") and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
            self.model.to(self.device)
            self.mode = "ml"
            print(
                f"✅ 護欄：ML + 平台放行（device={self.device}, "
                f"tokenizer={tok_src}, threshold={self.threshold}）"
            )
        except Exception as e:
            print(f"⚠️ 護欄 ML 未載入，改為規則模式（硬性規則 + 平台放行 + 輸出脫敏）：{e}")
            self.model = None
            self.tokenizer = None
            self.mode = "rules"

    def _hard_unsafe_hit(self, text: str) -> str | None:
        for pat in HARD_UNSAFE_PATTERNS:
            if pat.search(text):
                return pat.pattern
        return None

    def _is_platform_allowed(self, text: str) -> bool:
        """明確屬於本平台合規／診斷／報告用途 → 可覆寫 ML 誤擋。"""
        t = (text or "").strip()
        if not t:
            return False
        if _CASUAL_RE.match(t):
            return True
        if _CISCO_SYSLOG_RE.search(t):
            return True
        # 「我要的是 27001／iso27001」短句：ML 常誤判 unsafe=1.0
        if re.search(
            r"(iso\s*/?\s*iec\s*)?27001|iso\s*27001|iso27001|27002",
            t,
            re.I,
        ):
            return True
        if re.search(r"我(要的?是|是說|指的是|問的是)", t) and re.search(
            r"\d{4,5}|iso|合規|控制項",
            t,
            re.I,
        ):
            return True
        # 管理面加固／停用弱協議（「停用 Telnet」等常被 ML 誤擋）
        if _HARDENING_ALLOW_RE.search(t):
            return True
        # 產出／生成報告類（模型特別容易誤擋）
        if re.search(
            r"(產出|生成|產生|撰寫|寫|出).{0,16}(報告|診斷|稽核|合規)",
            t,
            re.I,
        ):
            return True
        if re.search(
            r"(報告|診斷|稽核).{0,12}(產出|生成|產生|撰寫|寫出|給我|幫我)",
            t,
            re.I,
        ):
            return True
        if re.search(
            r"(合規|診斷|稽核|iso|監控|日誌|syslog|控制項).{0,20}"
            r"(報告|分析|診斷|現況|風險|缺失)",
            t,
            re.I,
        ):
            return True
        # 其他平台關鍵字（至少命中一個領域詞）
        if _PLATFORM_ALLOW_RE.search(t):
            # 避免過寬：純「生成圖片」等非本域仍交給 ML
            if re.search(
                r"iso|ot|合規|稽核|診斷|報告|syslog|日誌|監控|radius|"
                r"控制項|修補|半導體|工控|a\.\d+|telnet|ssh|停用|禁用|"
                r"加固|步驟|設定|配置|27001|27002",
                t,
                re.I,
            ):
                return True
        return False

    def _ml_check(self, text: str) -> dict:
        import torch

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
        unsafe_prob = float(probs[1].item())
        label_id = 1 if unsafe_prob >= self.threshold else 0
        borderline_low = float(os.environ.get("GUARDRAIL_REVIEW_LOW", "0.35"))
        borderline_high = float(os.environ.get("GUARDRAIL_REVIEW_HIGH", "0.55"))
        human_review = (
            label_id == 0
            and borderline_low <= unsafe_prob <= borderline_high
        )
        result = {
            "label": ID2LABEL[label_id],
            "label_id": label_id,
            "safe_prob": float(probs[0].item()),
            "unsafe_prob": unsafe_prob,
            "blocked": label_id == 1,
            "mode": "ml",
            "matched_rules": 0,
            "reason": "微調護欄模型判定為不安全" if label_id == 1 else "微調護欄模型放行",
            "human_review_recommended": human_review,
            "human_review_reason": (
                f"邊界 unsafe 分數 {unsafe_prob:.3f}（{borderline_low}–{borderline_high}），"
                "建議 Guardrails Reviewer 覆核"
                if human_review
                else ""
            ),
            "guardrail_stack": self.mechanism_summary(),
        }
        return result

    def mechanism_summary(self) -> dict:
        """Guardrails 技術實施摘要（供 API／簡報）。"""
        if self.mode == "rules":
            return {
                "input_layers": [
                    "hard_rule_blocklist（注入／惡意／越獄）",
                    "platform_allowlist（OT／ISO／報告／加固意圖）",
                    "（ML 分類器未載入，Python/torch 環境限制時使用規則模式）",
                ],
                "output_layers": [
                    "IP／password／token 脫敏（REDACT_PATTERNS）",
                ],
                "human_review": [
                    "borderline ML 分數 → review_queue（需 ML 模式）",
                    "blocked 請求 → 分析師覆核",
                    "LLM 診斷 → 核准後方可作稽核證據",
                ],
                "phase2_options": ["NeMo Guardrails", "LlamaGuard 二道防線"],
                "threshold": self.threshold,
                "device": getattr(self, "device", "cpu"),
                "mode": self.mode,
            }
        return {
            "input_layers": [
                "hard_rule_blocklist（注入／惡意／越獄）",
                "platform_allowlist（OT／ISO／報告／加固意圖）",
                "fine_tuned RoBERTa 分類器（train_gur/fine_tuned_guardrail）",
            ],
            "output_layers": [
                "IP／password／token 脫敏（REDACT_PATTERNS）",
            ],
            "human_review": [
                "borderline ML 分數 → review_queue",
                "blocked 請求 → 分析師覆核",
                "LLM 診斷 → 核准後方可作稽核證據",
            ],
            "phase2_options": ["NeMo Guardrails", "LlamaGuard 二道防線"],
            "threshold": self.threshold,
            "device": getattr(self, "device", "cpu"),
            "mode": self.mode,
        }

    def check_input(self, text: str) -> dict:
        """ML 檢查 + 硬性惡意攔截 + 平台合法意圖放行。"""
        text = (text or "").strip()
        if not text:
            return {
                "label": "safe",
                "blocked": False,
                "safe_prob": 1.0,
                "unsafe_prob": 0.0,
                "mode": self.mode,
                "reason": "空輸入",
            }

        hard = self._hard_unsafe_hit(text)
        if hard:
            return {
                "label": "unsafe",
                "blocked": True,
                "safe_prob": 0.0,
                "unsafe_prob": 1.0,
                "mode": "hard-rule",
                "matched_rules": 1,
                "reason": f"觸發硬性安全規則：{hard}",
            }

        # 平台合法意圖：直接放行（不依賴可能誤判的 ML）
        if self._is_platform_allowed(text):
            return {
                "label": "safe",
                "blocked": False,
                "safe_prob": 1.0,
                "unsafe_prob": 0.0,
                "mode": "allowlist",
                "matched_rules": 0,
                "reason": "平台合規／診斷／報告／加固意圖，放行",
            }

        if self.mode != "ml" or self.model is None or self.tokenizer is None:
            reason = (
                "規則護欄放行（硬性規則 + 平台 allowlist；ML 未載入）"
                if self.mode == "rules"
                else "護欄模型未就緒，暫不攔截"
            )
            return {
                "label": "safe",
                "blocked": False,
                "safe_prob": 1.0,
                "unsafe_prob": 0.0,
                "mode": self.mode,
                "reason": reason,
            }

        try:
            return self._ml_check(text)
        except Exception as e:
            print(f"⚠️ 護欄 ML 推論失敗，暫不攔截：{e}")
            return {
                "label": "safe",
                "blocked": False,
                "safe_prob": 1.0,
                "unsafe_prob": 0.0,
                "mode": "ml-error",
                "reason": f"推論失敗：{e}",
            }

    def sanitize_output(self, text: str) -> tuple[str, bool]:
        """輸出脫敏，回傳 (文字, 是否有遮蔽)。"""
        if not text:
            return text, False
        redacted = text
        changed = False
        for pattern, repl in REDACT_PATTERNS:
            new_text, n = pattern.subn(repl, redacted)
            if n:
                changed = True
                redacted = new_text
        return redacted, changed

    def block_message(self, result: dict | None = None) -> str:
        reason = (result or {}).get("reason") or "觸發安全護欄"
        unsafe = (result or {}).get("unsafe_prob")
        extra = f"（unsafe={unsafe:.3f}）" if isinstance(unsafe, float) else ""
        return (
            "⚠️ 護欄已攔截此請求。\n"
            f"原因：{reason}{extra}\n"
            "本系統僅提供 ISO 27001 / OT 工控資安合規諮詢，"
            "請改以合法、與資安稽核相關的問題詢問。"
        )


guardrail_service = GuardrailService()
