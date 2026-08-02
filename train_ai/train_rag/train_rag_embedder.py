"""
微調 RAG 檢索用 Embedding 模型（Sentence Transformers）。

預設基底模型: BAAI/bge-base-zh-v1.5
  - 針對中文檢索優化，比 bge-m3 更輕、訓練更快
  - 適合 ISO 27001 / OT 繁中知識庫

用法:
  python prepare_rag_dataset.py
  python train_rag_embedder.py
  python train_rag_embedder.py --model BAAI/bge-small-zh-v1.5
  python build_index.py   # 訓練後重建索引
"""
import argparse
import json
import math
from pathlib import Path

from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
PAIRS_PATH = ROOT / "rag_pairs.jsonl"
OUTPUT_DIR = ROOT / "fine_tuned_rag_embedder"

# 中文檢索專用（推薦）
DEFAULT_MODEL = "BAAI/bge-base-zh-v1.5"
# 備選：
#   BAAI/bge-small-zh-v1.5   → 更省顯存
#   BAAI/bge-large-zh-v1.5   → 精度更高、較慢
#   BAAI/bge-m3              → 中英多語、模型較大


def load_pairs(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_pairs(rows, eval_ratio=0.15):
    n_eval = max(1, int(len(rows) * eval_ratio))
    return rows[n_eval:], rows[:n_eval]


def build_ir_evaluator(eval_rows):
    queries, corpus, relevant_docs = {}, {}, {}
    for i, row in enumerate(eval_rows):
        qid, did = f"q{i}", f"d{i}"
        queries[qid] = row["query"]
        corpus[did] = row["positive"]
        relevant_docs[qid] = {did}
    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="rag-eval",
        show_progress_bar=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Train Chinese RAG embedder")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"SentenceTransformer 基底模型（預設 {DEFAULT_MODEL}）",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if not PAIRS_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {PAIRS_PATH}，請先執行: python prepare_rag_dataset.py"
        )

    rows = load_pairs(PAIRS_PATH)
    train_rows, eval_rows = split_pairs(rows)
    print(f"訓練 pair: {len(train_rows)} | 評估 pair: {len(eval_rows)}")
    print(f"基底模型: {args.model}")

    model = SentenceTransformer(args.model)

    train_examples = [
        InputExample(texts=[r["query"], r["positive"]]) for r in train_rows
    ]
    batch_size = args.batch_size
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    evaluator = build_ir_evaluator(eval_rows)

    epochs = args.epochs
    steps_per_epoch = max(1, math.ceil(len(train_examples) / batch_size))
    warmup_steps = max(1, int(steps_per_epoch * epochs * 0.1))

    print("開始微調 RAG Embedding 模型...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        evaluation_steps=steps_per_epoch,
        output_path=str(OUTPUT_DIR),
        save_best_model=True,
        show_progress_bar=True,
    )

    print(f"微調完成，模型已儲存至: {OUTPUT_DIR}")
    print("請接著執行: python build_index.py")

    metrics = evaluator(model)
    print("\n========= RAG Embedding 評估 =========")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print("======================================")


if __name__ == "__main__":
    main()
