"""
RAG 檢索服務：向量 embedding（優先）或關鍵字檢索（後備）。
索引若不存在，啟動時會嘗試自動建立。
Embedding 預設放 CPU，避免與主 LLM 搶 GPU 顯存。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

try:
    import numpy as np
except Exception as _np_err:  # Python 3.14 等環境 numpy 可能無法載入
    np = None  # type: ignore[assignment]
    _NUMPY_IMPORT_ERROR = str(_np_err)
else:
    _NUMPY_IMPORT_ERROR = ""

from code.paths import train_ai_dir

RAG_ROOT = train_ai_dir() / "train_rag"
KB_PATH = RAG_ROOT / "knowledge_base.json"
EMBEDDER_DIR = RAG_ROOT / "fine_tuned_rag_embedder"
INDEX_DIR = RAG_ROOT / "index"
# 無微調模型時的後備：中文檢索專用（比 bge-m3 更輕）
FALLBACK_EMBEDDER = "BAAI/bge-base-zh-v1.5"


def _env_rag_enabled() -> bool:
    return os.environ.get("OT_ENABLE_RAG", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _doc_text(doc: dict) -> str:
    """相容 text/content 與 instruction/output 兩種知識庫格式。"""
    text = (doc.get("text") or doc.get("content") or "").strip()
    if text:
        return text
    instruction = (doc.get("instruction") or doc.get("title") or "").strip()
    output = (doc.get("output") or "").strip()
    if instruction and output:
        return f"問題: {instruction}\n答案: {output}"
    return output or instruction


class RagService:
    def __init__(self, top_k: int = 3, min_score: float = 0.25):
        self.top_k = top_k
        self.min_score = min_score
        self.enabled = False
        self.mode = "off"  # vector | keyword | off
        self.embedder = None
        self.embeddings = None
        self.docs: list[dict] = []
        self._init()

    def _init(self):
        if not _env_rag_enabled():
            print("📚 RAG：功能已關閉（OT_ENABLE_RAG=0）")
            return
        if not KB_PATH.exists():
            print(f"📚 RAG：找不到知識庫 {KB_PATH}，RAG 停用。")
            return

        if np is None:
            print(f"📚 RAG：numpy 不可用（{_NUMPY_IMPORT_ERROR[:80]}…），改用關鍵字檢索")
            self._init_keyword_mode()
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("📚 RAG：未安裝 sentence-transformers，改用關鍵字檢索")
            self._init_keyword_mode()
            return

        model_path = str(EMBEDDER_DIR) if EMBEDDER_DIR.exists() else FALLBACK_EMBEDDER
        try:
            print(f"📚 RAG：載入 Embedding（CPU）: {model_path}")
            self.embedder = SentenceTransformer(model_path, device="cpu")
        except Exception as e:
            print(f"⚠️ RAG Embedding 載入失敗，改用關鍵字檢索：{e}")
            self._init_keyword_mode()
            return

        try:
            if not self._index_ready():
                print("📚 RAG：索引不存在，正在自動建立（首次可能較久）...")
                self._build_index()
            self.embeddings, self.docs = self._load_index()
            if not self._dims_compatible():
                print(
                    f"📚 RAG：索引維度 {getattr(self.embeddings, 'shape', None)} "
                    f"與目前 Embedding 不符，重新建立索引..."
                )
                self._build_index()
                self.embeddings, self.docs = self._load_index()
            if not self._dims_compatible():
                raise RuntimeError(
                    f"索引維度仍不相容：index={self.embeddings.shape}"
                )
            self.mode = "vector"
            self.enabled = True
            print(f"✅ RAG：就緒（向量），文件數={len(self.docs)}，向量={self.embeddings.shape}")
        except Exception as e:
            print(f"⚠️ RAG 向量索引失敗，改用關鍵字檢索：{e}")
            self._init_keyword_mode()

    def _init_keyword_mode(self):
        """無 numpy／embedding 時：直接讀知識庫做關鍵字檢索。"""
        try:
            with open(KB_PATH, "r", encoding="utf-8") as f:
                self.docs = json.load(f) or []
            if not self.docs:
                print("📚 RAG：knowledge_base 為空，RAG 停用。")
                return
            self.embedder = None
            self.embeddings = None
            self.mode = "keyword"
            self.enabled = True
            print(f"✅ RAG：就緒（關鍵字），文件數={len(self.docs)}")
        except Exception as e:
            print(f"⚠️ RAG 關鍵字模式失敗，RAG 停用：{e}")
            self.enabled = False

    def _index_ready(self) -> bool:
        return (INDEX_DIR / "embeddings.npy").exists() and (INDEX_DIR / "meta.json").exists()

    def _probe_embed_dim(self) -> int | None:
        if self.embedder is None or np is None:
            return None
        try:
            v = self.embedder.encode(
                ["dimension probe"],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            arr = np.asarray(v, dtype=np.float32)
            if arr.ndim == 1:
                return int(arr.shape[0])
            if arr.ndim >= 2:
                return int(arr.shape[-1])
        except Exception as e:
            print(f"⚠️ RAG：無法探測 Embedding 維度：{e}")
        return None

    def _dims_compatible(self) -> bool:
        if self.embeddings is None or np is None:
            return False
        emb = np.asarray(self.embeddings)
        if emb.ndim != 2 or emb.shape[0] == 0:
            return False
        dim = self._probe_embed_dim()
        if dim is None:
            return True
        return int(emb.shape[1]) == int(dim)

    def _build_index(self):
        if self.embedder is None or np is None:
            raise RuntimeError("無法建立向量索引（缺少 embedder 或 numpy）")
        with open(KB_PATH, "r", encoding="utf-8") as f:
            docs = json.load(f)
        texts = [_doc_text(d) for d in docs]
        if not docs or not any(t.strip() for t in texts):
            raise RuntimeError("knowledge_base 為空或缺少可用文字欄位，無法建立索引")
        embeddings = self.embedder.encode(
            texts,
            batch_size=16,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        np.save(INDEX_DIR / "embeddings.npy", embeddings)
        with open(INDEX_DIR / "meta.json", "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)
        print(f"📚 RAG：索引已寫入 {embeddings.shape}")

    def _load_index(self):
        if np is None:
            raise RuntimeError("numpy 不可用")
        embeddings = np.load(INDEX_DIR / "embeddings.npy")
        with open(INDEX_DIR / "meta.json", "r", encoding="utf-8") as f:
            docs = json.load(f)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings, docs

    @staticmethod
    def _extract_cisco_keys(text: str) -> list[str]:
        keys = []
        for m in re.finditer(
            r"%?([A-Z][A-Z0-9_-]+)-(\d+)-([A-Z][A-Z0-9_]+)",
            text or "",
            re.I,
        ):
            fac, sev, mne = m.group(1).upper(), m.group(2), m.group(3).upper()
            keys.append(f"%{fac}-{sev}-{mne}")
            keys.append(f"%{fac}-{mne}")
            keys.append(mne)
        t = (text or "").upper()
        for alias in (
            "LOGIN_SUCCESS", "LOGIN_FAILED", "LOGOUT", "TTY_EXPIRE",
            "AUTHFAIL", "CONFIG_I", "MALWARE", "RADIUS", "SNMP",
        ):
            if alias in t:
                keys.append(alias)
        seen = set()
        out = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out

    @staticmethod
    def _extract_query_terms(query: str) -> list[str]:
        """關鍵字檢索用：Cisco 碼、Annex A、英文詞、中文片段。"""
        q = query or ""
        terms: list[str] = []
        for m in re.finditer(r"%?[A-Z][A-Z0-9_-]+-\d+-[A-Z0-9_]+", q, re.I):
            terms.append(m.group(0).upper())
        for m in re.finditer(r"A\.\d+(?:\.\d+)?", q, re.I):
            terms.append(m.group(0).upper())
        for m in re.finditer(r"ISO\s*/?\s*IEC?\s*2700[0-5]", q, re.I):
            terms.append(re.sub(r"\s+", " ", m.group(0)).upper())
        for m in re.finditer(r"[A-Za-z][A-Za-z0-9_.-]{2,}", q):
            terms.append(m.group(0).lower())
        for m in re.finditer(r"[\u4e00-\u9fff]{2,12}", q):
            seg = m.group(0)
            terms.append(seg)
            if len(seg) >= 4:
                for i in range(len(seg) - 1):
                    terms.append(seg[i : i + 2])
        seen = set()
        out = []
        for t in terms:
            key = t.lower()
            if key not in seen and len(t.strip()) >= 2:
                seen.add(key)
                out.append(t)
        return out

    def _doc_blob(self, item: dict) -> str:
        return " ".join(
            str(item.get(x) or "")
            for x in ("instruction", "output", "text", "title", "tags", "content")
        )

    def _apply_hit_boosts(self, item: dict, blob_upper: str, cisco_keys: list[str]) -> float:
        boost = 0.0
        src = (item.get("source") or "").lower()
        if src in ("ot_log", "curated") or item.get("doc_type") == "log_analysis":
            boost += 0.12
        if src in ("train_json", "old_kb_filtered"):
            boost -= 0.08
        for ck in cisco_keys:
            if ck.upper() in blob_upper:
                boost += 0.18
                break
        if re.search(r"LOGIN_SUCCESS|SEC_LOGIN", blob_upper) and re.search(
            r"重放攻擊|REPLAY ATTACK", blob_upper
        ):
            boost -= 0.35
        return boost

    def _finalize_hit(self, item: dict, adj: float, base: float) -> dict:
        answer = (item.get("output") or item.get("content") or "").strip()
        content = answer or _doc_text(item)
        if len(content) > 900:
            content = content[:900] + "...(已截斷)"
        item = dict(item)
        item["snippet"] = content
        item["score"] = adj
        item["score_raw"] = base
        if not item.get("title"):
            item["title"] = (item.get("instruction") or item.get("id") or "")[:120]
        return item

    def _retrieve_keyword(self, query: str, top_k: int) -> list[dict]:
        k = max(top_k or self.top_k, 3)
        pool = max(k * 8, 24)
        cisco_keys = self._extract_cisco_keys(query)
        terms = self._extract_query_terms(query)
        if not terms:
            terms = [query.strip()[:40]] if query.strip() else []

        scored: list[tuple[float, float, dict]] = []
        for doc in self.docs:
            blob = self._doc_blob(doc)
            blob_upper = blob.upper()
            blob_lower = blob.lower()
            raw = 0.0
            title_upper = (
                f"{doc.get('title') or ''} {doc.get('instruction') or ''}"
            ).upper()

            for term in terms:
                tu = term.upper()
                tl = term.lower()
                if tu in blob_upper or tl in blob_lower:
                    raw += 1.2 if len(term) >= 4 else 0.7
                if tu in title_upper:
                    raw += 1.0

            if raw <= 0:
                continue

            boost = self._apply_hit_boosts(doc, blob_upper, cisco_keys)
            # 映射到 0–1 分數，與向量檢索過濾邏輯相容
            base = min(0.92, 0.18 + raw * 0.08)
            adj = min(0.98, base + boost)
            if adj < self.min_score and base < self.min_score:
                continue
            scored.append((adj, base, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [
            self._finalize_hit(dict(doc), adj, base)
            for adj, base, doc in scored[:pool]
        ]
        return results[:k]

    def _retrieve_vector(self, query: str, top_k: int) -> list[dict]:
        if self.embedder is None or self.embeddings is None or np is None:
            return []
        k = max(top_k or self.top_k, 3)
        pool = max(k * 6, 18)
        cisco_keys = self._extract_cisco_keys(query)

        q = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        q = np.asarray(q, dtype=np.float32)
        q_vec = q if q.ndim == 1 else q[0]

        emb = np.asarray(self.embeddings, dtype=np.float32)
        if emb.ndim != 2:
            emb = emb.reshape(len(self.docs), -1)

        if emb.shape[1] != q_vec.shape[0]:
            print(
                f"⚠️ RAG：檢索維度不符 index={emb.shape} query={q_vec.shape}，重建索引..."
            )
            self._build_index()
            self.embeddings, self.docs = self._load_index()
            emb = np.asarray(self.embeddings, dtype=np.float32)
            if emb.shape[1] != q_vec.shape[0]:
                print("❌ RAG：重建後維度仍不符，略過本次檢索")
                return []

        scores = (emb @ q_vec).tolist()
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:pool]

        results = []
        for idx, score in ranked:
            if idx < 0 or idx >= len(self.docs):
                continue
            base = float(score)
            item = dict(self.docs[idx])
            blob_upper = self._doc_blob(item).upper()
            boost = self._apply_hit_boosts(item, blob_upper, cisco_keys)
            adj = base + boost
            if adj < self.min_score and base < self.min_score:
                continue
            results.append(self._finalize_hit(item, adj, base))

        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results[:k]

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        if not self.enabled or not query or not self.docs:
            return []
        try:
            if self.mode == "keyword":
                return self._retrieve_keyword(query, top_k or self.top_k)
            if self.mode == "vector":
                return self._retrieve_vector(query, top_k or self.top_k)
            return []
        except Exception as e:
            print(f"⚠️ RAG retrieve 失敗（已略過，不中斷對話）：{e}")
            return []

    def _analysis_answer(self, hit: dict, limit: int = 420) -> str:
        text = (hit.get("output") or hit.get("content") or hit.get("snippet") or "").strip()
        if not text:
            text = _doc_text(hit)
        text = re.sub(r"^問題:.*?\n答案:\s*", "", text, flags=re.S)
        text = re.sub(r"\**場域情境（[^）)]{1,40}）\**[：:][^\n]*", "", text)
        text = re.sub(
            r"汽車組裝廠|汽車廠|食品廠|鋼鐵廠|石化廠|水泥廠|紙漿廠|"
            r"天然氣調壓站|港口碼頭|風力發電場|火力電廠|充電場站|製藥廠",
            "半導體廠",
            text,
        )
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) > limit:
            text = text[:limit].rstrip() + "…"
        return text

    def format_context(self, hits: list[dict], max_chars: int = 900) -> str:
        """給 LLM：同類 syslog／合規知識的標準分析答案。"""
        if not hits:
            return ""
        blocks = []
        used = 0
        for i, h in enumerate(hits, 1):
            tags = h.get("tags") or []
            tag_hint = ""
            if isinstance(tags, list) and tags:
                tag_hint = f"（{'/'.join(str(t) for t in tags[:3])}）"
            title = str(h.get("title") or h.get("id") or f"doc_{i}")[:60]
            m = re.search(r"%([A-Z0-9_-]+)-(\d+)-([A-Z0-9_]+)", title, re.I)
            if m:
                title = f"%{m.group(1)}-{m.group(2)}-{m.group(3)} 專業參考"
            answer = self._analysis_answer(h)
            if not answer:
                continue
            block = f"【專業參考 {i}{tag_hint}｜{title}】\n{answer}"
            if used + len(block) > max_chars:
                if not blocks:
                    blocks.append(block[:max_chars])
                break
            blocks.append(block)
            used += len(block) + 2
        return "\n\n".join(blocks)

    def citation_payload(self, hits: list[dict]) -> list[dict]:
        out = []
        for h in hits:
            doc_id = h.get("id") or h.get("doc_id") or ""
            out.append({
                "evidence_id": f"RAG-{doc_id}" if doc_id else "",
                "id": doc_id,
                "title": (h.get("title") or h.get("instruction") or "")[:120],
                "score": round(float(h.get("score") or 0), 3),
                "source": h.get("source") or "knowledge_base",
                "mode": self.mode,
            })
        return out


rag_service = RagService(top_k=3, min_score=0.22)
