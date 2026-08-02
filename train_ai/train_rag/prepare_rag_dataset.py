"""
把 train.json（instruction/output）轉成 RAG 微調用的 pair 資料。

產出:
  - rag_pairs.jsonl  : SentenceTransformers 訓練用 (query, positive)
  - knowledge_base.json : 檢索知識庫文件（以回答為 chunk）
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN_AI_DIR = ROOT.parent
PROJECT_ROOT = TRAIN_AI_DIR.parent
# 訓練資料在 train_ai/train_llm/train.json（相容舊路徑 train_ai/train.json）
_SRC_CANDIDATES = [
    TRAIN_AI_DIR / "train_llm" / "train.json",
    TRAIN_AI_DIR / "train.json",
    PROJECT_ROOT / "train.json",
]
SRC_PATH = next((p for p in _SRC_CANDIDATES if p.is_file()), _SRC_CANDIDATES[0])
PAIRS_PATH = ROOT / "rag_pairs.jsonl"
KB_PATH = ROOT / "knowledge_base.json"
SEED = 42


def main():
    if not SRC_PATH.exists():
        raise FileNotFoundError(
            f"找不到來源資料: {SRC_PATH}\n"
            f"請確認存在：{TRAIN_AI_DIR / 'train_llm' / 'train.json'}"
        )

    with open(SRC_PATH, "r", encoding="utf-8") as f:
        rows = json.load(f)

    pairs = []
    knowledge = []
    for i, row in enumerate(rows):
        query = row["instruction"].strip()
        answer = row["output"].strip()
        doc_id = f"doc_{i:04d}"

        # 知識庫文件：同時保留 instruction/output（現有 KB 格式）與 text（索引用）
        knowledge.append(
            {
                "id": doc_id,
                "title": query,
                "content": answer,
                "text": f"問題: {query}\n答案: {answer}",
                "instruction": query,
                "output": answer,
            }
        )
        pairs.append({"query": query, "positive": answer})

    random.seed(SEED)
    random.shuffle(pairs)

    with open(PAIRS_PATH, "w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(KB_PATH, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, ensure_ascii=False, indent=2)

    print(f"來源：{SRC_PATH}")
    print(f"已寫入 {len(pairs)} pairs → {PAIRS_PATH}")
    print(f"已寫入 {len(knowledge)} docs → {KB_PATH}")


if __name__ == "__main__":
    main()
