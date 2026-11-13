# -*- coding: utf-8 -*-
"""Run text2sql_day1 pipeline against local Ollama base Gemma 4 E4B (not OT-finetuned)."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

BASE_URL = "http://127.0.0.1:11434/v1"
API_KEY = "ollama"
# Native QAT base; override with TEXT2SQL_MODEL if needed (e.g. gemma4_e4b_ot).
MODEL = os.environ.get("TEXT2SQL_MODEL", "gemma4:e4b-it-qat")
N_QUESTIONS = int(os.environ.get("N_QUESTIONS", "30"))

import glob

import text2sql_common as tsc

DATA_DIR = "minidev_data"
_SAMPLE_CACHE: dict[str, str] = {}


def find(name):
    hits = [
        p
        for p in glob.glob(f"{DATA_DIR}/**/{name}", recursive=True)
        if "__MACOSX" not in p
    ]
    if not hits:
        raise FileNotFoundError(f"Cannot find {name} under {DATA_DIR}")
    return sorted(hits, key=len)[0]


QUESTIONS_FILE = find("mini_dev_sqlite.json")
DB_ROOT = find("dev_databases")


def db_path(db_id):
    return os.path.join(DB_ROOT, db_id, f"{db_id}.sqlite")


def get_schema(db_id):
    conn = sqlite3.connect(db_path(db_id))
    rows = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
    ).fetchall()
    conn.close()
    return "\n\n".join(r[0] for r in rows)


def _sample_values(db_id: str) -> str:
    if db_id not in _SAMPLE_CACHE:
        _SAMPLE_CACHE[db_id] = tsc.get_sample_cell_values(db_path(db_id))
    return _SAMPLE_CACHE[db_id]


def build_prompt(item, use_evidence=True):
    ev = item.get("evidence") if use_evidence else None
    return tsc.build_prompt_text(
        schema_ddl=get_schema(item["db_id"]),
        question=item["question"],
        evidence=ev or None,
        sample_values=_sample_values(item["db_id"]),
    )


def ask_llm(prompt, retries=3):
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + "/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.load(resp)
            return data["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "ignore")[:500]
            if e.code == 429 and attempt < retries - 1:
                time.sleep(10 * (attempt + 1))
                continue
            raise RuntimeError(f"API 錯誤 {e.code}：{msg}") from None


extract_sql = tsc.extract_sql


def generate_pred_sql(item, use_evidence=True, max_attempts=3) -> str:
    prompt = build_prompt(item, use_evidence)
    last_sql = ""
    extra = ""
    for attempt in range(max_attempts):
        reply = ask_llm(prompt + extra)
        last_sql = extract_sql(reply)
        ok, err = tsc.sql_parse_ok(db_path(item["db_id"]), last_sql)
        if ok:
            return last_sql
        if attempt + 1 >= max_attempts:
            break
        if tsc.sql_needs_retry(err, last_sql):
            extra = (
                "\n\n【修正】上一個 SQL 無法解析（"
                + err[:160]
                + "）。請重新輸出完整單行 SQL：括號與引號成對、含必要 FROM/JOIN、"
                "Date/Country/Currency 格式與「欄位抽樣」一致。"
                + (f"\n錯誤的 SQL：{last_sql[:600]}" if last_sql else "")
            )
        else:
            break
    return last_sql


def run_sql(db_id, sql, timeout_s=30):
    conn = sqlite3.connect(db_path(db_id))
    deadline = time.time() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10000)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def is_correct(db_id, pred_sql, gold_sql):
    try:
        pred = run_sql(db_id, pred_sql)
    except Exception as e:
        return False, f"模型的 SQL 執行失敗：{e}"
    gold = run_sql(db_id, gold_sql)
    return set(pred) == set(gold), ""


def main():
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)
    subset = questions[:N_QUESTIONS]

    print(f"MODEL={MODEL} BASE_URL={BASE_URL} N={len(subset)}")
    print(f"QUESTIONS_FILE={QUESTIONS_FILE}")

    item0 = subset[0]
    t0 = time.time()
    reply0 = ask_llm(build_prompt(item0))
    print(f"smoke_ok elapsed={time.time()-t0:.1f}s sql_len={len(extract_sql(reply0))}")

    records_with = []
    records_without = []
    for i, item in enumerate(subset, 1):
        for use_ev, bucket, label in (
            (True, records_with, "with_evidence"),
            (False, records_without, "without_evidence"),
        ):
            try:
                pred_sql = generate_pred_sql(item, use_ev)
                ok, err = is_correct(item["db_id"], pred_sql, item["SQL"])
            except RuntimeError as e:
                pred_sql, ok, err = "", False, str(e)
            bucket.append(
                {
                    "idx": i,
                    "db_id": item["db_id"],
                    "difficulty": item.get("difficulty", ""),
                    "question": item["question"],
                    "evidence": item.get("evidence", ""),
                    "pred_sql": pred_sql,
                    "gold_sql": item["SQL"],
                    "correct": ok,
                    "error": err,
                }
            )
        done = sum(r["correct"] for r in records_with)
        print(f"\rprogress {i}/{len(subset)} correct(with_ev)={done}", end="", flush=True)
    print()

    ex_with = sum(r["correct"] for r in records_with) / len(records_with) * 100
    ex_without = sum(r["correct"] for r in records_without) / len(records_without) * 100

    out = {
        "model": MODEL,
        "base_url": BASE_URL,
        "n_questions": len(subset),
        "ex_with_evidence_pct": round(ex_with, 2),
        "ex_without_evidence_pct": round(ex_without, 2),
        "gap_pp": round(ex_with - ex_without, 2),
        "with_evidence": records_with,
        "without_evidence": records_without,
    }
    out_path = ROOT / "scripts" / "text2sql_gemma_e4b_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"EX with evidence: {ex_with:.1f}%")
    print(f"EX without evidence: {ex_without:.1f}%")
    print(f"gap: {ex_with - ex_without:+.1f} pp")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
