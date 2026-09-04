"""
輸出幻覺偵測：微調 RoBERTa 二分類（grounded / hallucinated）。

推論輸入格式：[Q] 問題 [E] 證據 [A] 回答

環境變數：
  ENABLE_HALLUCINATION_GUARD=1   啟用（預設與 ENABLE_GUARDRAIL 相同）
  HALLUCINATION_THRESHOLD=0.6   超過視為幻覺
  HALLUCINATION_DEVICE=cpu
  HALLUCINATION_BASE_TOKENIZER=hfl/chinese-roberta-wwm-ext
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from code.paths import train_ai_dir

MODEL_DIR = train_ai_dir() / "train_hallucination" / "fine_tuned_hallucination_guard"
BASE_TOKENIZER_ID = os.environ.get(
    "HALLUCINATION_BASE_TOKENIZER", "hfl/chinese-roberta-wwm-ext"
)

ID2LABEL = {0: "grounded", 1: "hallucinated"}
SEP_Q = " [Q] "
SEP_E = " [E] "
SEP_A = " [A] "


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class HallucinationGuardService:
    def __init__(self, threshold: float | None = None):
        env_th = os.environ.get("HALLUCINATION_THRESHOLD")
        if threshold is not None:
            self.threshold = float(threshold)
        elif env_th:
            self.threshold = float(env_th)
        else:
            self.threshold = 0.6
        self.mode = "disabled"
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self._init_model()

    def _enabled(self) -> bool:
        if os.environ.get("ENABLE_HALLUCINATION_GUARD") is not None:
            return _env_flag("ENABLE_HALLUCINATION_GUARD", default=True)
        return _env_flag("ENABLE_GUARDRAIL", default=True)

    def _tokenizer_is_usable(self, tokenizer) -> bool:
        try:
            if len(tokenizer) < 1000:
                return False
            sample = f"{SEP_Q}合規現況{SEP_E}count=12{SEP_A}fail 12 筆"
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
            tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
            if self._tokenizer_is_usable(tok):
                return tok, "local"
        except Exception as e:
            print(f"⚠️ 幻覺護欄：本地 tokenizer 失敗：{e}")

        tok = AutoTokenizer.from_pretrained(BASE_TOKENIZER_ID)
        if not self._tokenizer_is_usable(tok):
            raise RuntimeError("幻覺護欄基底 tokenizer 不可用")
        return tok, "base"

    def _init_model(self):
        if not self._enabled():
            print("🔍 幻覺護欄：已停用（ENABLE_HALLUCINATION_GUARD=0）")
            self.mode = "disabled"
            return
        if not MODEL_DIR.exists() or not (MODEL_DIR / "config.json").is_file():
            print(f"⚠️ 幻覺護欄：找不到模型 {MODEL_DIR}，改規則模式")
            self.mode = "rules"
            return
        if not (MODEL_DIR / "model.safetensors").is_file() and not list(
            MODEL_DIR.glob("pytorch_model*.bin")
        ):
            print(f"⚠️ 幻覺護欄：缺少權重檔，改規則模式")
            self.mode = "rules"
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification

            self.tokenizer, tok_src = self._load_tokenizer()
            self.model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
            self.model.eval()
            pref = (os.environ.get("HALLUCINATION_DEVICE") or "cuda").strip().lower()
            if pref in ("cuda", "gpu") and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
            self.model.to(self.device)
            self.mode = "ml"
            print(
                f"✅ 幻覺護欄：ML（device={self.device}, "
                f"tokenizer={tok_src}, threshold={self.threshold}）"
            )
        except Exception as e:
            print(f"⚠️ 幻覺護欄 ML 未載入，改規則模式：{e}")
            self.model = None
            self.tokenizer = None
            self.mode = "rules"

    @staticmethod
    def _format_input(question: str, evidence: str, answer: str) -> str:
        q = (question or "").strip()
        e = (evidence or "").strip() or "（無證據）"
        a = (answer or "").strip()
        if len(e) > 1200:
            e = e[:1200] + "…"
        if len(a) > 800:
            a = a[:800] + "…"
        return f"{SEP_Q}{q}{SEP_E}{e}{SEP_A}{a}"

    def _ml_check(self, question: str, evidence: str, answer: str) -> dict:
        import torch

        text = self._format_input(question, evidence, answer)
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=384,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
        hall_prob = float(probs[1].item())
        is_hall = hall_prob >= self.threshold
        return {
            "label": ID2LABEL[1 if is_hall else 0],
            "label_id": 1 if is_hall else 0,
            "grounded_prob": float(probs[0].item()),
            "hallucination_prob": hall_prob,
            "is_hallucination": is_hall,
            "mode": "ml",
            "reason": "ML 判定回答可能含幻覺" if is_hall else "ML 判定回答有依據",
        }

    @staticmethod
    def _rules_check(question: str, evidence: str, answer: str) -> dict | None:
        """規則後備：與 app._report_hallucinates_against_log 對齊的輕量版。"""
        r = answer or ""
        ev = evidence or ""
        if not r.strip():
            return None

        bad_patterns = [
            r"MFG\s*\d+|MFG01|P01\s*機臺",
            r"\[HOSTNAME\]|\[IP_ADDRESS\]|SYSLOG:\s*\[ID\s*\d+\]",
            r"新增(?:新)?設定檔|刪除舊設定檔",
            r"事件類型\s*[:：]\s*資安報告|生成報告狀態\s*[:：]",
            r"22007|25701|27101|2900\s*系列",
            r"network_monitor\s+\d+",
            r"12345|45678|99999",
            r"不要補劇本|防幻覺鐵律|智慧合規診斷報告",
            r"使用者[：:].*assistant[：:]",
        ]
        for pat in bad_patterns:
            if re.search(pat, r, re.I):
                if pat.startswith("MFG") and re.search(r"MFG|P01", ev, re.I):
                    continue
                return {
                    "label": "hallucinated",
                    "label_id": 1,
                    "grounded_prob": 0.0,
                    "hallucination_prob": 1.0,
                    "is_hallucination": True,
                    "mode": "rules",
                    "reason": f"規則偵測幻覺特徵：{pat[:40]}",
                }

        if ev and not re.search(r"%[A-Z0-9_-]+-\d+-[A-Z0-9_]+", ev, re.I):
            if re.search(r"%[A-Z0-9_-]+-\d+-[A-Z0-9_]+", r, re.I) and "本次日誌" in r:
                return {
                    "label": "hallucinated",
                    "label_id": 1,
                    "grounded_prob": 0.0,
                    "hallucination_prob": 1.0,
                    "is_hallucination": True,
                    "mode": "rules",
                    "reason": "規則：證據無 syslog 卻捏造訊息碼",
                }
        return None

    def check_output(
        self,
        question: str,
        answer: str,
        *,
        evidence: str = "",
        ot_context: str = "",
        rag_context: str = "",
    ) -> dict:
        """檢查 LLM 回答是否可能含幻覺。"""
        if self.mode == "disabled":
            return {
                "label": "grounded",
                "is_hallucination": False,
                "grounded_prob": 1.0,
                "hallucination_prob": 0.0,
                "mode": "disabled",
                "reason": "幻覺護欄已停用",
            }

        ans = (answer or "").strip()
        if not ans:
            return {
                "label": "grounded",
                "is_hallucination": False,
                "grounded_prob": 1.0,
                "hallucination_prob": 0.0,
                "mode": self.mode,
                "reason": "空回答",
            }

        ev = "\n".join(x for x in (evidence, ot_context, rag_context) if x).strip()

        rule_hit = self._rules_check(question, ev, ans)
        if rule_hit:
            return rule_hit

        if self.mode != "ml" or self.model is None or self.tokenizer is None:
            return {
                "label": "grounded",
                "is_hallucination": False,
                "grounded_prob": 1.0,
                "hallucination_prob": 0.0,
                "mode": self.mode,
                "reason": "規則未命中且 ML 未載入，放行",
            }

        try:
            return self._ml_check(question, ev, ans)
        except Exception as e:
            print(f"⚠️ 幻覺護欄推論失敗：{e}")
            return {
                "label": "grounded",
                "is_hallucination": False,
                "grounded_prob": 1.0,
                "hallucination_prob": 0.0,
                "mode": "ml-error",
                "reason": str(e),
            }

    def mechanism_summary(self) -> dict:
        return {
            "mode": self.mode,
            "threshold": self.threshold,
            "device": getattr(self, "device", "cpu"),
            "model_dir": str(MODEL_DIR),
            "input_format": f"{SEP_Q}question{SEP_E}evidence{SEP_A}answer",
            "labels": ID2LABEL,
        }


hallucination_guard_service = HallucinationGuardService()
