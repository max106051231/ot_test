"""
輸出幻覺偵測模型微調：二分類 Sequence Classification
0 = grounded, 1 = hallucinated

用法:
  python generate_hallucination_dataset.py
  python train_hallucination.py
  python train_hallucination.py --epochs 3
"""
from __future__ import annotations

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
DATA_PATH = ROOT / "hallucination_dataset.json"
CHECKPOINT_DIR = ROOT / "hallucination_checkpoints"
OUTPUT_DIR = ROOT / "fine_tuned_hallucination_guard"

DEFAULT_MODEL = "hfl/chinese-roberta-wwm-ext"
LABEL2ID = {"grounded": 0, "hallucinated": 1}
ID2LABEL = {0: "grounded", 1: "hallucinated"}

SEP_Q = " [Q] "
SEP_E = " [E] "
SEP_A = " [A] "


def format_sample(item: dict) -> str:
    q = (item.get("question") or "").strip()
    e = (item.get("evidence") or "").strip() or "（無證據）"
    a = (item.get("answer") or "").strip()
    return f"{SEP_Q}{q}{SEP_E}{e}{SEP_A}{a}"


class HallucinationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=384):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = format_sample(item)
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def load_tokenizer(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)


def build_training_args(epochs: int):
    common = dict(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
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
    parser = argparse.ArgumentParser(description="Train hallucination guard classifier")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到 {DATA_PATH}，請先執行 generate_hallucination_dataset.py")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    labels = [d["label"] for d in raw]
    train_data, test_data = train_test_split(
        raw, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"訓練集: {len(train_data)} | 測試集: {len(test_data)} | 基底: {args.model}")

    tokenizer = load_tokenizer(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    trainer = Trainer(
        model=model,
        args=build_training_args(args.epochs),
        train_dataset=HallucinationDataset(train_data, tokenizer),
        eval_dataset=HallucinationDataset(test_data, tokenizer),
        compute_metrics=compute_metrics,
    )

    print("開始訓練幻覺偵測模型…")
    trainer.train()

    results = trainer.evaluate()
    print("\n========= 測試集 =========")
    print(f"Accuracy : {results['eval_accuracy']:.4f}")
    print(f"Precision: {results['eval_precision']:.4f}")
    print(f"Recall   : {results['eval_recall']:.4f}")
    print(f"F1       : {results['eval_f1']:.4f}")
    print("==========================")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    meta = {
        "format": f"{SEP_Q}question{SEP_E}evidence{SEP_A}answer",
        "base_model": args.model,
        "label2id": LABEL2ID,
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
    }
    with open(OUTPUT_DIR / "hallucination_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"已儲存至 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
