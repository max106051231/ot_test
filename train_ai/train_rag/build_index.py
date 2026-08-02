"""
用微調後的 Embedding 模型，對知識庫建立向量索引。

產出:
  - index/embeddings.npy
  - index/meta.json
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
KB_PATH = ROOT / "knowledge_base.json"
EMBEDDER_DIR = ROOT / "fine_tuned_rag_embedder"
INDEX_DIR = ROOT / "index"
# 與 train_rag_embedder.py 預設一致；若已有微調模型則優先用微調版
FALLBACK_MODEL = "BAAI/bge-base-zh-v1.5"


def main():
    if not KB_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {KB_PATH}，請先執行: python prepare_rag_dataset.py"
        )

    model_path = EMBEDDER_DIR if EMBEDDER_DIR.exists() else FALLBACK_MODEL
    print(f"載入 Embedding 模型: {model_path}")
    model = SentenceTransformer(str(model_path))

    with open(KB_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)

    def doc_text(d: dict) -> str:
        text = (d.get("text") or d.get("content") or "").strip()
        if text:
            return text
        instruction = (d.get("instruction") or d.get("title") or "").strip()
        output = (d.get("output") or "").strip()
        if instruction and output:
            return f"問題: {instruction}\n答案: {output}"
        return output or instruction

    texts = [doc_text(d) for d in docs]
    if not any(texts):
        raise RuntimeError("knowledge_base 缺少可用文字欄位（text/content 或 instruction/output）")
    print(f"正在編碼 {len(texts)} 篇文件...")
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", embeddings)
    with open(INDEX_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"索引已建立: {INDEX_DIR}")
    print(f"向量形狀: {embeddings.shape}")


if __name__ == "__main__":
    main()
