"""
簡易 RAG 問答：檢索微調後的工控知識庫，再交給生成模型回答。

用法:
  python prepare_rag_dataset.py
  python train_rag_embedder.py
  python build_index.py
  python rag_chat.py
  python rag_chat.py --question "什麼是 Modbus TCP 功能碼 03？"
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
TRAIN_AI_DIR = ROOT.parent
PROJECT_ROOT = TRAIN_AI_DIR.parent
INDEX_DIR = ROOT / "index"
EMBEDDER_DIR = ROOT / "fine_tuned_rag_embedder"
FALLBACK_EMBEDDER = "BAAI/bge-base-zh-v1.5"
# 本地 LLM 候選（優先 train_ai/models → train_llm → 根目錄舊路徑）
_LLM_CANDIDATES = [
    TRAIN_AI_DIR / "models" / "phi4_merged_model",
    TRAIN_AI_DIR / "models" / "qwen_ot_merged_model",
    TRAIN_AI_DIR / "train_llm" / "phi4_merged_model",
    TRAIN_AI_DIR / "train_llm" / "qwen_ot_merged_model",
    PROJECT_ROOT / "qwen_ot_merged_model",
    TRAIN_AI_DIR / "phi4_merged_model",
]
DEFAULT_LLM = next(
    (p for p in _LLM_CANDIDATES if (p / "config.json").is_file()),
    _LLM_CANDIDATES[2],
)
FALLBACK_LLM = "Qwen/Qwen2.5-3B-Instruct"


def load_index():
    emb_path = INDEX_DIR / "embeddings.npy"
    meta_path = INDEX_DIR / "meta.json"
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"找不到索引，請先執行: python build_index.py\n預期路徑: {INDEX_DIR}"
        )
    embeddings = np.load(emb_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return embeddings, docs


def retrieve(query, embedder, embeddings, docs, top_k=3):
    q = embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    scores = (embeddings @ q[0]).tolist()
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for idx, score in ranked:
        item = dict(docs[idx])
        item["score"] = float(score)
        results.append(item)
    return results


def _ctx_text(c: dict) -> str:
    text = (c.get("text") or c.get("content") or "").strip()
    if text:
        return text
    instruction = (c.get("instruction") or c.get("title") or "").strip()
    output = (c.get("output") or "").strip()
    if instruction and output:
        return f"問題: {instruction}\n答案: {output}"
    return output or instruction


def build_prompt(question: str, contexts: list[dict]) -> str:
    ctx_blocks = []
    for i, c in enumerate(contexts, 1):
        ctx_blocks.append(
            f"[文件{i} | score={c['score']:.3f}]\n{_ctx_text(c)}"
        )
    context_text = "\n\n".join(ctx_blocks)
    user_msg = (
        "請根據下列檢索到的工控知識回答問題。"
        "若知識不足以回答，請明確說明無法從資料推得答案，不要捏造。\n\n"
        f"【檢索內容】\n{context_text}\n\n"
        f"【問題】\n{question}"
    )
    return (
        "<|system|>\n你是工控與 OT 資安領域的專業助手，"
        "回答需依據提供的檢索內容，條理清楚。<|end|>\n"
        f"<|user|>\n{user_msg}<|end|>\n"
        "<|assistant|>\n"
    )


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens=768):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
    )
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="RAG chat for OT knowledge")
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--llm", type=str, default="")
    parser.add_argument("--embedder", type=str, default="")
    args = parser.parse_args()

    embedder_path = args.embedder or (
        str(EMBEDDER_DIR) if EMBEDDER_DIR.exists() else FALLBACK_EMBEDDER
    )
    llm_path = args.llm or (
        str(DEFAULT_LLM)
        if (Path(DEFAULT_LLM) / "config.json").is_file()
        else FALLBACK_LLM
    )

    print(f"Embedding: {embedder_path}")
    print(f"LLM      : {llm_path}")

    embedder = SentenceTransformer(embedder_path)
    embeddings, docs = load_index()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(llm_path, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        token=token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def answer_one(question: str):
        hits = retrieve(question, embedder, embeddings, docs, top_k=args.top_k)
        print("\n--- 檢索結果 ---")
        for i, h in enumerate(hits, 1):
            title = h.get("title") or h.get("instruction") or h.get("id") or ""
            print(f"{i}. score={h['score']:.3f} | {str(title)[:80]}")
        prompt = build_prompt(question, hits)
        reply = generate(model, tokenizer, prompt)
        print("\n--- RAG 回答 ---")
        print(reply)
        return reply

    if args.question:
        answer_one(args.question)
        return

    print("\nRAG 問答終端（輸入 exit 離開）")
    while True:
        q = input("\n問題: ").strip()
        if q.lower() in {"exit", "quit", "q"}:
            break
        if not q:
            continue
        answer_one(q)


if __name__ == "__main__":
    main()
