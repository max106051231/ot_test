"""
護欄模型微調：二分類 Sequence Classification
0 = Safe, 1 = Unsafe

用法:
  python train_Guard.py
  python train_Guard.py --model hfl/chinese-macbert-base

說明:
  預設使用 hfl/chinese-roberta-wwm-ext：
  - 中文分類效果優於 bert-base-chinese
  - WordPiece tokenizer，避開 Deberta-v3 / SentencePiece 相容問題
  - 適合護欄 Safe/Unsafe 二分類與繁中資安語句
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "guardrail_dataset.json"
CHECKPOINT_DIR = ROOT / "guardrail_checkpoints"
OUTPUT_DIR = ROOT / "fine_tuned_guardrail"

# 中文序列分類較佳基底（整詞遮罩 RoBERTa）
DEFAULT_MODEL = "hfl/chinese-roberta-wwm-ext"
# 備選：
#   hfl/chinese-macbert-base
#   bert-base-chinese
#   hfl/chinese-bert-wwm-ext
LABEL2ID = {"safe": 0, "unsafe": 1}
ID2LABEL = {0: "safe", 1: "unsafe"}


class GuardrailDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def load_tokenizer(model_name: str):
    """優先 fast tokenizer；失敗再退回 slow。"""
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e1:
        print(f"⚠️ fast tokenizer 失敗，改試 slow：{e1}")
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)


def build_training_args():
    """相容不同 transformers 版本的參數名稱。"""
    common = dict(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=str(ROOT / "logs"),
        logging_steps=5,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=42,
    )
    try:
        return TrainingArguments(eval_strategy="epoch", **common)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", **common)


def main():
    parser = argparse.ArgumentParser(description="Train guardrail classifier")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace 模型名稱（預設 {DEFAULT_MODEL}）",
    )
    args = parser.parse_args()
    model_name = args.model

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {DATA_PATH}，請先執行: python generate_dataset.py"
        )

    print("正在載入並切分資料集...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    labels = [d["label"] for d in raw_data]
    train_data, test_data = train_test_split(
        raw_data,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    print(f"訓練集: {len(train_data)} 筆 | 測試集: {len(test_data)} 筆")
    print(f"基底模型: {model_name}")

    tokenizer = load_tokenizer(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    train_dataset = GuardrailDataset(train_data, tokenizer)
    test_dataset = GuardrailDataset(test_data, tokenizer)
    training_args = build_training_args()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("開始訓練護欄模型...")
    trainer.train()

    print("\n正在測試集上進行最終評估...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print("\n========= 測試集評估結果 =========")
    print(f"Accuracy : {test_results['eval_accuracy']:.4f}")
    print(f"Precision: {test_results['eval_precision']:.4f}")
    print(f"Recall   : {test_results['eval_recall']:.4f}")
    print(f"F1 Score : {test_results['eval_f1']:.4f}")
    print("====================================")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n正在將最佳模型儲存至: {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("儲存完成！之後重啟 app.py 即可自動載入 ML 護欄。")


if __name__ == "__main__":
    main()
