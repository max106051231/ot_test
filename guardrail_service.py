"""
護欄服務：純 ML 輸入攔截 + 輸出脫敏。
使用 train_ai/train_gur/fine_tuned_guardrail，不做規則式 Safe/Unsafe 判斷。

環境變數：
  GUARDRAIL_THRESHOLD=0.5
  GUARDRAIL_BASE_TOKENIZER=hfl/chinese-roberta-wwm-ext
"""
from __future__ import annotations

import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
GUARD_DIR = BASE_DIR / "train_ai" / "train_gur" / "fine_tuned_guardrail"
BASE_TOKENIZER_ID = os.environ.get(
    "GUARDRAIL_BASE_TOKENIZER", "hfl/chinese-roberta-wwm-ext"
)

ID2LABEL = {0: "safe", 1: "unsafe"}

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
        if not GUARD_DIR.exists() or not (GUARD_DIR / "config.json").is_file():
            print(f"❌ 護欄：找不到微調模型 {GUARD_DIR}，輸入檢查將全部放行。")
            self.mode = "disabled"
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification

            self.tokenizer, tok_src = self._load_tokenizer()
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(GUARD_DIR)
            )
            self.model.eval()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.mode = "ml"
            print(
                f"✅ 護欄：純 ML 模式（device={self.device}, "
                f"tokenizer={tok_src}, threshold={self.threshold}）"
            )
        except Exception as e:
            print(f"❌ 護欄模型載入失敗，輸入檢查將全部放行：{e}")
            self.model = None
            self.tokenizer = None
            self.mode = "disabled"

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
        return {
            "label": ID2LABEL[label_id],
            "label_id": label_id,
            "safe_prob": float(probs[0].item()),
            "unsafe_prob": unsafe_prob,
            "blocked": label_id == 1,
            "mode": "ml",
            "matched_rules": 0,
            "reason": "微調護欄模型判定為不安全" if label_id == 1 else "微調護欄模型放行",
        }

    def check_input(self, text: str) -> dict:
        """純 ML 檢查使用者輸入是否安全（無規則式）。"""
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

        if self.mode != "ml" or self.model is None or self.tokenizer is None:
            return {
                "label": "safe",
                "blocked": False,
                "safe_prob": 1.0,
                "unsafe_prob": 0.0,
                "mode": self.mode,
                "reason": "護欄模型未就緒，暫不攔截",
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
