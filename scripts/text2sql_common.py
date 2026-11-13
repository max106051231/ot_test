# -*- coding: utf-8 -*-
"""Shared Text2SQL prompt helpers, sample values, and SQL normalization."""
from __future__ import annotations

import re
import sqlite3

SCHEMA_CONVENTIONS = """### 資料庫慣例（無論是否有 evidence 都必須遵守）
- 僅使用 CREATE TABLE 中存在的表與欄位；JOIN 前先確認欄位在哪張表（勿把 A 表欄位寫到 B 表別名上）。
- 欄位 Date 若抽樣為 6 位字串（如 201308）：表示 YYYYMM，請用 Date='201308'、SUBSTR(Date,1,4)='2013'、SUBSTR(Date,5,2) 等，不要擅自用 '2013-08-01' / strftime，除非抽樣已是 ISO 日期。
- transactions_1k.Date 若抽樣為 '2012-08-24'：該表用 ISO 日期；yearmonth 等表仍可能是 YYYYMM，不可混用格式。
- Country、Currency 等代碼欄：只用抽樣中的實際值（如 CZE、SVK、EUR、CZK），禁止 Czech Republic、euro 等自然語言。
- 客戶分級 SME/LAM/KAM 通常在 customers.Segment；加油站類型 Premium/Discount 在 gasstations.Segment，兩者不可混淆。
- transactions_1k 常同時有 Amount（數量）與 Price（金額）；題目問 spent/paid/price 時優先用 Price。
- 凡 SELECT 用到表欄位，必須有對應 FROM/JOIN；字串常數須完整閉合（'CZK' 不可寫成 'CZ 或 'CZ)）。"""


def get_sample_cell_values(db_path: str) -> str:
    """每表抽 1～2 筆 Date/Country/Currency/Segment 等，供模型對齊取值格式。"""
    want = ("Date", "Country", "Currency", "Segment", "Time")
    lines: list[str] = []
    conn = sqlite3.connect(db_path)
    try:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        ]
        for table in sorted(tables):
            cols = [
                r[1]
                for r in conn.execute(f"PRAGMA table_info({table})").fetchall()
            ]
            hit = [c for c in cols if c in want]
            if not hit:
                continue
            sel = ", ".join(hit)
            try:
                rows = conn.execute(
                    f"SELECT {sel} FROM {table} WHERE "
                    + " OR ".join(f"{c} IS NOT NULL" for c in hit)
                    + " LIMIT 2"
                ).fetchall()
            except sqlite3.Error:
                continue
            for row in rows:
                pairs = [f"{c}={row[i]!r}" for i, c in enumerate(hit)]
                lines.append(f"- {table}: " + ", ".join(pairs))
    finally:
        conn.close()
    if not lines:
        return "（無抽樣）"
    return "\n".join(lines[:24])


def build_prompt_text(
    *,
    schema_ddl: str,
    question: str,
    evidence: str | None,
    sample_values: str,
) -> str:
    parts = [
        "你是 SQLite 專家。根據下面的資料庫結構，寫出能回答問題的 SQLite 查詢。",
        "輸出規則（必須全部遵守）：",
        "1. 只輸出一條 SQL 語句，禁止任何前後贅詞、問候、解釋、Markdown、程式碼區塊標記。",
        "2. SQL 必須寫在同一行，禁止換行（不要 \\n）。",
        "3. 第一個字元必須是 SELECT 或 WITH（依需要）。",
        "4. 所有括號、CASE/END、字串引號必須完整閉合；整句必須可直接在 SQLite 執行。",
        "",
        SCHEMA_CONVENTIONS,
        "",
        "### 資料庫結構",
        schema_ddl,
        "",
        "### 欄位抽樣（database content，請對齊格式）",
        sample_values,
        "",
        "### 問題",
        question,
    ]
    if evidence:
        parts += ["", "### 提示（外部知識）", evidence]
    return "\n".join(parts)


def sql_literals_balanced(sql: str) -> bool:
    in_single = False
    i = 0
    while i < len(sql):
        if sql[i] == "'":
            if in_single:
                if i + 1 < len(sql) and sql[i + 1] == "'":
                    i += 2
                    continue
                in_single = False
            else:
                in_single = True
        i += 1
    return not in_single


def balance_closing_parens(sql: str) -> str:
    depth = 0
    in_single = False
    i = 0
    while i < len(sql):
        c = sql[i]
        if in_single:
            if c == "'":
                if i + 1 < len(sql) and sql[i + 1] == "'":
                    i += 1
                else:
                    in_single = False
        elif c == "'":
            in_single = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth = max(0, depth - 1)
        i += 1
    if depth > 0:
        sql = sql + ")" * depth
    return sql


def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    sql = m.group(1) if m else text
    sql = sql.strip()
    for prefix in ("SQL:", "sql:", "Query:", "query:", "Answer:", "答案：", "查询："):
        if sql.startswith(prefix):
            sql = sql[len(prefix) :].strip()
    m2 = re.search(r"\b(WITH|SELECT)\b.*", sql, re.S | re.I)
    if m2:
        sql = m2.group(0)
    sql = re.sub(r"\s+", " ", sql).strip().rstrip(";").strip()
    return balance_closing_parens(sql)


def sql_needs_retry(err: str, sql: str) -> bool:
    err_l = (err or "").lower()
    if not sql_literals_balanced(sql):
        return True
    if "incomplete input" in err_l:
        return True
    if "unrecognized token" in err_l and ("'" in err or err_l.count("'") % 2 == 1):
        return True
    if "unbalanced" in err_l or "syntax error" in err_l:
        return True
    if "no such column" in err_l and not re.search(r"\bFROM\b", sql, re.I):
        return True
    return False


def sql_parse_ok(db_path: str, sql: str) -> tuple[bool, str]:
    if not sql or not re.match(r"^\s*(WITH|SELECT)\b", sql, re.I):
        return False, "empty or not SELECT/WITH"
    if not sql_literals_balanced(sql):
        return False, "unclosed string literal in SQL"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(f"EXPLAIN {sql}")
        return True, ""
    except sqlite3.Error as e:
        return False, str(e)
    finally:
        conn.close()
