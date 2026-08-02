"""
RAG 檢索服務：使用微調後的 embedding + knowledge_base 向量索引。
索引若不存在，啟動時會嘗試自動建立。
Embedding 預設放 CPU，避免與主 LLM 搶 GPU 顯存。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
RAG_ROOT = BASE_DIR / "train_ai" / "train_rag"
KB_PATH = RAG_ROOT / "knowledge_base.json"
EMBEDDER_DIR = RAG_ROOT / "fine_tuned_rag_embedder"
INDEX_DIR = RAG_ROOT / "index"
# 無微調模型時的後備：中文檢索專用（比 bge-m3 更輕）
FALLBACK_EMBEDDER = "BAAI/bge-base-zh-v1.5"


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
        self.embedder = None
        self.embeddings = None
        self.docs = []
        self._init()

    def _init(self):
        if not KB_PATH.exists():
            print(f"📚 RAG：找不到知識庫 {KB_PATH}，RAG 停用。")
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("📚 RAG：未安裝 sentence-transformers，RAG 停用。請 pip install sentence-transformers")
            return

        model_path = str(EMBEDDER_DIR) if EMBEDDER_DIR.exists() else FALLBACK_EMBEDDER
        try:
            print(f"📚 RAG：載入 Embedding（CPU）: {model_path}")
            self.embedder = SentenceTransformer(model_path, device="cpu")
        except Exception as e:
            print(f"⚠️ RAG Embedding 載入失敗，RAG 停用：{e}")
            return

        try:
            if not self._index_ready():
                print("📚 RAG：索引不存在，正在自動建立（首次可能較久）...")
                self._build_index()
            self.embeddings, self.docs = self._load_index()
            # 舊索引維度與目前模型不一致時自動重建（常見：1024→768）
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
            self.enabled = True
            print(f"✅ RAG：就緒，文件數={len(self.docs)}，向量={self.embeddings.shape}")
        except Exception as e:
            print(f"⚠️ RAG 索引準備失敗，RAG 停用：{e}")
            self.enabled = False

    def _index_ready(self) -> bool:
        return (INDEX_DIR / "embeddings.npy").exists() and (INDEX_DIR / "meta.json").exists()

    def _probe_embed_dim(self) -> int | None:
        """偵測目前 embedder 輸出維度。"""
        if self.embedder is None:
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
        if self.embeddings is None:
            return False
        emb = np.asarray(self.embeddings)
        if emb.ndim != 2 or emb.shape[0] == 0:
            return False
        dim = self._probe_embed_dim()
        if dim is None:
            return True
        return int(emb.shape[1]) == int(dim)

    def _build_index(self):
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
        embeddings = np.load(INDEX_DIR / "embeddings.npy")
        with open(INDEX_DIR / "meta.json", "r", encoding="utf-8") as f:
            docs = json.load(f)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings, docs

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        if not self.enabled or not query or self.embedder is None:
            return []
        if self.embeddings is None or not len(self.docs):
            return []

        try:
            k = top_k or self.top_k
            q = self.embedder.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            q = np.asarray(q, dtype=np.float32)
            if q.ndim == 1:
                q_vec = q
            else:
                q_vec = q[0]

            emb = np.asarray(self.embeddings, dtype=np.float32)
            if emb.ndim != 2:
                emb = emb.reshape(len(self.docs), -1)

            # 維度漂移時嘗試重建一次
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
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

            results = []
            for idx, score in ranked:
                if idx < 0 or idx >= len(self.docs):
                    continue
                if float(score) < self.min_score:
                    continue
                item = dict(self.docs[idx])
                item["score"] = float(score)
                content = _doc_text(item)
                if len(content) > 700:
                    content = content[:700] + "...(已截斷)"
                item["snippet"] = content
                if not item.get("title"):
                    item["title"] = (item.get("instruction") or item.get("id") or "")[:120]
                results.append(item)
            return results
        except Exception as e:
            print(f"⚠️ RAG retrieve 失敗（已略過，不中斷對話）：{e}")
            return []

    def _summarize_snippet(self, snippet: str, limit: int = 160) -> str:
        """把知識片段壓成短摘要，去掉原始 Log 行，降低模型照抄機率。"""
        if not snippet:
            return ""
        text = str(snippet)
        # 去掉典型 syslog / 時間戳日誌行
        text = re.sub(
            r"(?m)^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\b.*$",
            "",
            text,
        )
        text = re.sub(r"(?m)^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}.*$", "", text)
        text = re.sub(r"(?m)^(debug|info|warn|error|trace)[:\s].*$", "", text, flags=re.I)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > limit:
            text = text[:limit].rstrip() + "…"
        return text

    def format_context(self, hits: list[dict], max_chars: int = 700) -> str:
        """給 LLM 的精簡參考（條列重點，不含原始 Log 全文）。"""
        if not hits:
            return ""
        blocks = []
        used = 0
        for i, h in enumerate(hits, 1):
            title = (h.get("title") or h.get("id") or f"doc_{i}")[:80]
            summary = self._summarize_snippet(h.get("snippet") or "")
            if not summary:
                summary = "（僅標題可參考，無可用摘要）"
            block = f"{i}. {title}：{summary}"
            if used + len(block) > max_chars:
                break
            blocks.append(block)
            used += len(block)
        return "\n".join(blocks)

    def citation_payload(self, hits: list[dict]) -> list[dict]:
        """回傳給前端顯示的精簡引用（不含 Log 原文）。"""
        out = []
        for h in hits:
            out.append({
                "id": h.get("id"),
                "title": (h.get("title") or "")[:120],
                "score": round(float(h.get("score") or 0), 3),
            })
        return out


rag_service = RagService(top_k=3, min_score=0.22)
