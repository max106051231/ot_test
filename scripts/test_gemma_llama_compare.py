#!/usr/bin/env python3
"""Gemma vs Llama 微調前／微調後四組模型對照測試與報告。"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "http://127.0.0.1:2000"
ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "scripts" / "gemma_llama_compare_results.json"
OUT_MD = ROOT / "docs" / "GEMMA_LLAMA_COMPARE_REPORT.md"

# 四組模型 slug 偏好（依 ollama_models.json）
TARGETS = {
    "gemma_base": ["gemma2:2b", "base_gemma2_2b"],
    "gemma_finetuned": ["gemma_2b_ot", "gemma2b_ot"],
    "llama_base": ["llama3.2:3b", "base:meta-llama/Llama-3.2-3B-Instruct"],
    "llama_finetuned": ["llama32_3b_ot"],
}

CHAT_CASES = [
    ("casual", "你好", "寒暄"),
    ("iso27001", "你知道 iso27001 是什麼嗎", "知識問答"),
    ("compliance", "目前合規現況如何？", "合規現況"),
    ("patch", "給我修補步驟", "修補步驟"),
    ("hardening", "如何停用 Telnet 並改用 SSH？", "加固指引"),
    ("chart", "用圓餅圖看各控制項事件占比", "圖表"),
    ("syslog", "%SEC_LOGIN-4-LOGIN_FAILED: Login failed from 10.0.0.5", "日誌分析"),
    ("off_topic", "今天台東天氣如何？", "離題"),
]

BAD_PATTERNS = [
    (r"Ollama 推論失敗|模型輸出異常|輸出異常", "llm_error"),
    (r"\[UNK_BYTE_", "gemma_corrupt"),
    (r"peg-native format|Ollama HTTP 500", "ollama_crash"),
    (r"Security Techniques|Information Security Management Systems", "english_bullet"),
    (r"^Please provide me with some context", "english_only"),
    (r"分析使用者|使用者本則問題", "meta_leak"),
]


def http(method: str, path: str, body: dict | None = None, timeout: float = 240) -> tuple[int, object]:
    url = f"{BASE}{path}"
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def cjk_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text or ""))


def analyze_reply(reply: str, stage: str) -> list[str]:
    issues: list[str] = []
    text = reply or ""
    if not text.strip():
        issues.append("empty_reply")
    if len(text.strip()) < 8:
        issues.append("too_short")
    for pat, tag in BAD_PATTERNS:
        if re.search(pat, text, re.I | re.M):
            issues.append(tag)
    if not re.search(r"[\u4e00-\u9fff]", text) and len(text) > 40:
        issues.append("no_chinese")
    if re.search(r"信息|软件|网络|数据|组织|登录|账户", text):
        issues.append("simplified_chinese")
    if stage == "base" and re.search(
        r"Security Techniques|Information Security Management", text, re.I
    ):
        issues.append("base_rag_leak")
    # 合規現況：base 常泛化
    return issues


def resolve_targets(all_models: list[dict]) -> dict[str, dict | None]:
    by_slug = {}
    by_path = {}
    for m in all_models:
        by_slug[m.get("slug") or ""] = m
        by_path[m.get("path") or ""] = m
        by_path[m.get("load_path") or ""] = m

    resolved: dict[str, dict | None] = {}
    for key, prefs in TARGETS.items():
        found = None
        for p in prefs:
            if p in by_slug:
                found = by_slug[p]
                break
            if p in by_path:
                found = by_path[p]
                break
        if not found:
            # fuzzy: gemma/llama + stage
            fam = "gemma" if "gemma" in key else "llama"
            stage = "base" if "base" in key else "finetuned"
            for m in all_models:
                slug = (m.get("slug") or "").lower()
                path = (m.get("path") or "").lower()
                st = m.get("stage") or ""
                if fam not in slug and fam not in path:
                    continue
                if stage == "base" and st != "base":
                    continue
                if stage != "base" and st not in ("finetuned", "alias"):
                    continue
                if m.get("available", True):
                    found = m
                    break
        resolved[key] = found
    return resolved


def switch_model(slug: str) -> tuple[bool, str]:
    _, data = http("POST", "/api/llm/switch", {"model": slug})
    if not isinstance(data, dict) or not data.get("ok"):
        err = data.get("error") if isinstance(data, dict) else str(data)
        return False, err or "switch failed"
    return True, data.get("label") or slug


def run_model_group(group_key: str, model: dict) -> list[dict]:
    slug = model.get("slug") or model.get("path") or ""
    stage = model.get("stage") or "unknown"
    label = model.get("label") or slug
    family = "Gemma" if "gemma" in group_key else "Llama"
    tuning = "微調前" if "base" in group_key else "微調後"

    if not model.get("available", True):
        return [{
            "group": group_key,
            "family": family,
            "tuning": tuning,
            "model_label": label,
            "stage": stage,
            "case": "switch",
            "ok": False,
            "error": model.get("unavailable_reason") or "模型不可用",
        }]

    ok, msg = switch_model(slug)
    if not ok:
        return [{
            "group": group_key,
            "family": family,
            "tuning": tuning,
            "model_label": label,
            "stage": stage,
            "case": "switch",
            "ok": False,
            "error": msg,
        }]

    rows = []
    history: list[dict] = []
    for case_id, message, category in CHAT_CASES:
        t0 = time.time()
        _, data = http("POST", "/api/agent/chat", {
            "message": message,
            "history": history[-4:],
        })
        elapsed = round(time.time() - t0, 1)
        if not isinstance(data, dict):
            rows.append({
                "group": group_key, "family": family, "tuning": tuning,
                "model_label": msg, "stage": stage, "case": case_id,
                "category": category, "ok": False, "error": str(data),
            })
            continue

        status = data.get("status")
        reply = data.get("reply") or ""
        citations = len(data.get("rag_citations") or [])
        issues = analyze_reply(reply, stage) if status == "success" else [f"status_{status}"]

        if stage == "base" and citations > 0 and case_id in ("iso27001", "syslog", "chart"):
            issues.append("base_has_rag")
        if stage in ("finetuned", "alias") and case_id == "iso27001" and citations == 0:
            issues.append("missing_rag")

        # 合規：finetuned 應提及控制項或 fail
        if case_id == "compliance" and status == "success":
            if stage in ("finetuned", "alias") and not re.search(
                r"A\.\d|fail|pass|review|控制", reply, re.I
            ):
                issues.append("compliance_not_grounded")
            if stage == "base" and re.search(
                r"提供更多|具體的組織|視情況", reply
            ):
                issues.append("compliance_generic")

        rows.append({
            "group": group_key,
            "family": family,
            "tuning": tuning,
            "model_label": msg,
            "stage": stage,
            "case": case_id,
            "category": category,
            "ok": status == "success" and not issues,
            "status": status,
            "elapsed": elapsed,
            "issues": issues,
            "reply_len": len(reply),
            "reply_preview": re.sub(r"\s+", " ", reply)[:200],
            "rag_citations": citations,
            "tool": data.get("tool_name"),
            "cjk_chars": cjk_count(reply),
        })
        if status == "success" and reply:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": reply[:500]})
    return rows


def summarize(rows: list[dict]) -> dict[str, dict]:
    groups: dict[str, list] = {}
    for r in rows:
        if r.get("case") == "switch":
            continue
        groups.setdefault(r.get("group") or "?", []).append(r)

    out = {}
    for g, rs in groups.items():
        total = len(rs)
        passed = sum(1 for x in rs if x.get("ok"))
        out[g] = {
            "family": rs[0].get("family") if rs else "",
            "tuning": rs[0].get("tuning") if rs else "",
            "label": rs[0].get("model_label") if rs else g,
            "passed": passed,
            "total": total,
            "pass_rate": round(100 * passed / max(total, 1), 1),
            "avg_elapsed": round(sum(x.get("elapsed") or 0 for x in rs) / max(total, 1), 2),
            "total_rag": sum(x.get("rag_citations") or 0 for x in rs),
            "avg_reply_len": round(sum(x.get("reply_len") or 0 for x in rs) / max(total, 1)),
            "issues": list(dict.fromkeys(
                i for x in rs for i in (x.get("issues") or [])
            )),
        }
    return out


def pros_cons(group_key: str, s: dict) -> tuple[list[str], list[str]]:
    is_base = "base" in group_key
    is_gemma = "gemma" in group_key
    pros, cons = [], []

    if is_base:
        pros += [
            "無 RAG／底稿注入，延遲較低",
            "適合評估官方基底能力",
        ]
        cons += [
            "ISO/合規知識較弱，易泛化回答",
            "無知識圖譜引用",
        ]
    else:
        pros += [
            "Semi-Shield OT 微調 + RAG + 底稿",
            "合規現況可對照監控計數",
            "知識題具 RAG 引用",
        ]
        cons += [
            "延遲與 token 成本較高",
        ]

    if is_gemma:
        pros.append("2B 參數量小，理論上推理最快")
        cons += [
            "Gemma 2B 容量小，長回答易截斷或格式異常",
            "歷史上易出現 UNK/peg-native 類 Ollama 錯誤",
        ]
    else:
        pros += [
            "3B 參數，ISO/OT 敘述通常比 2B 完整",
            "Llama 3.2 Instruct 指令遵循較穩",
        ]
        cons.append("模型略大，比 Gemma 2B 占資源")

    if s.get("pass_rate", 0) >= 85:
        pros.append(f"本次通過率 {s['pass_rate']}%")
    else:
        cons.append(f"本次通過率僅 {s['pass_rate']}%")
    if s.get("avg_elapsed", 0) <= 1.0:
        pros.append(f"平均 {s['avg_elapsed']}s/題")
    elif s.get("avg_elapsed", 0) >= 2.0:
        cons.append(f"平均 {s['avg_elapsed']}s/題，偏慢")

    return pros, cons


def render_md(payload: dict) -> str:
    ts = payload["timestamp"]
    summary = payload.get("summary", {})
    rows = payload.get("rows", [])
    targets = payload.get("targets", {})

    lines = [
        "# Gemma vs Llama 微調前／微調後 對照測試報告",
        "",
        f"- **測試時間**：{ts}",
        f"- **後端**：{BASE}",
        "",
        "## 1. 測試模型",
        "",
        "| 分組 | 家族 | 階段 | 模型 | 可用 |",
        "|------|------|------|------|------|",
    ]
    order = ["gemma_base", "gemma_finetuned", "llama_base", "llama_finetuned"]
    labels = {
        "gemma_base": ("Gemma", "微調前"),
        "gemma_finetuned": ("Gemma", "微調後"),
        "llama_base": ("Llama", "微調前"),
        "llama_finetuned": ("Llama", "微調後"),
    }
    for g in order:
        m = targets.get(g)
        fam, tun = labels[g]
        if m:
            avail = "✅" if m.get("available", True) else "❌"
            lines.append(f"| {g} | {fam} | {tun} | {m.get('label')} (`{m.get('slug')}`) | {avail} |")
        else:
            lines.append(f"| {g} | {fam} | {tun} | *未找到* | ❌ |")

    lines += ["", "## 2. 總覽比較", "", "| 分組 | 通過率 | 平均耗時 | RAG 引用 | 平均字數 |", "|------|--------|----------|----------|----------|"]
    for g in order:
        s = summary.get(g, {})
        if not s:
            lines.append(f"| {g} | - | - | - | - |")
        else:
            lines.append(
                f"| {s.get('label', g)} | {s.get('pass_rate')}% ({s.get('passed')}/{s.get('total')}) | "
                f"{s.get('avg_elapsed')}s | {s.get('total_rag')} | {s.get('avg_reply_len')} |"
            )

    lines += ["", "## 3. 家族內：微調前 vs 微調後", ""]
    for fam, base_k, fine_k in [
        ("Gemma 2B", "gemma_base", "gemma_finetuned"),
        ("Llama 3.2 3B", "llama_base", "llama_finetuned"),
    ]:
        b, f = summary.get(base_k, {}), summary.get(fine_k, {})
        lines += [f"### {fam}", ""]
        if b and f:
            lines += [
                f"- **微調前** {b.get('label')}：通過 {b.get('pass_rate')}%，RAG {b.get('total_rag')}，均時 {b.get('avg_elapsed')}s",
                f"- **微調後** {f.get('label')}：通過 {f.get('pass_rate')}%，RAG {f.get('total_rag')}，均時 {f.get('avg_elapsed')}s",
                f"- **RAG 增益**：{f.get('total_rag', 0) - b.get('total_rag', 0)} 次引用",
                f"- **速度差**：微調後比微調前慢約 {round((f.get('avg_elapsed') or 0) - (b.get('avg_elapsed') or 0), 2)}s/題",
                "",
            ]

    lines += ["## 4. 跨家族：同阶段对比", "", "### 微調前 Gemma vs Llama", ""]
    gb, lb = summary.get("gemma_base", {}), summary.get("llama_base", {})
    if gb and lb:
        faster = "Gemma" if (gb.get("avg_elapsed") or 99) < (lb.get("avg_elapsed") or 99) else "Llama"
        lines += [
            f"- 速度：{faster} 較快（Gemma {gb.get('avg_elapsed')}s vs Llama {lb.get('avg_elapsed')}s）",
            f"- 通过率：Gemma {gb.get('pass_rate')}% vs Llama {lb.get('pass_rate')}%",
            "",
        ]
    lines += ["### 微調後 Gemma OT vs Llama OT", ""]
    gf, lf = summary.get("gemma_finetuned", {}), summary.get("llama_finetuned", {})
    if gf and lf:
        lines += [
            f"- RAG：Gemma OT {gf.get('total_rag')} vs Llama OT {lf.get('total_rag')}",
            f"- 通过率：Gemma OT {gf.get('pass_rate')}% vs Llama OT {lf.get('pass_rate')}%",
            f"- 均时：Gemma OT {gf.get('avg_elapsed')}s vs Llama OT {lf.get('avg_elapsed')}s",
            "",
        ]

    lines += ["## 5. 各模型優缺點", ""]
    for g in order:
        s = summary.get(g, {})
        if not s:
            continue
        p, c = pros_cons(g, s)
        lines += [f"### {s.get('label')}（{s.get('tuning')}）", "", "**優點**", ""]
        for x in p:
            lines.append(f"- {x}")
        lines += ["", "**缺點**", ""]
        for x in c:
            lines.append(f"- {x}")
        if s.get("issues"):
            lines += ["", f"**本次問題**：{', '.join(s['issues'])}", ""]
        lines.append("")

    lines += ["## 6. 關鍵案例並排", ""]
    for case_id, _, cat in CHAT_CASES:
        if case_id not in ("iso27001", "compliance", "syslog"):
            continue
        lines += [f"### {cat}（{case_id}）", ""]
        for g in order:
            hit = next((r for r in rows if r.get("group") == g and r.get("case") == case_id), None)
            if not hit:
                continue
            mark = "✅" if hit.get("ok") else "❌"
            lines.append(
                f"- {mark} **{hit.get('tuning')} {hit.get('family')}** "
                f"({hit.get('elapsed')}s, RAG={hit.get('rag_citations', 0)})"
            )
            lines.append(f"  > {hit.get('reply_preview', '')}")
        lines.append("")

    lines += [
        "## 7. 完整結果表",
        "",
        "| 模型 | 阶段 | 案例 | 结果 | 耗时 | RAG | 字数 | 问题 |",
        "|------|------|------|------|------|-----|------|------|",
    ]
    for r in rows:
        if r.get("case") == "switch":
            continue
        mark = "✅" if r.get("ok") else "❌"
        iss = ", ".join(r.get("issues") or []) or "-"
        lines.append(
            f"| {r.get('model_label', '')[:18]} | {r.get('tuning')} | {r.get('case')} | {mark} | "
            f"{r.get('elapsed', '-')}s | {r.get('rag_citations', 0)} | {r.get('reply_len', 0)} | {iss} |"
        )

    lines += [
        "",
        "## 8. 结论建议",
        "",
        "1. **OT/ISO 合規諮詢**：優先 **微調後 Llama 3.2 3B OT**（3B + RAG + 底稿，通過率與 grounded 度通常最佳）。",
        "2. **資源極度受限**：可試 **Gemma 2B OT**，但需接受較短回答與 occasional Ollama 異常。",
        "3. **對照實驗／基底評測**：用 **gemma2:2b** / **llama3.2:3b** 微調前，預期無 RAG、合規現況偏泛化。",
        "4. 複測：`python scripts/test_gemma_llama_compare.py`",
        "",
        "---",
        f"*自動產生於 {ts}*",
    ]
    return "\n".join(lines)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=== Gemma vs Llama Compare Test ===\n")
    _, safety = http("GET", "/api/safety/status")
    if not isinstance(safety, dict):
        print("FAIL: server unreachable")
        return 1

    _, models_resp = http("GET", "/api/llm/models")
    all_models = (models_resp.get("models") if isinstance(models_resp, dict) else []) or []
    targets = resolve_targets(all_models)

    for k, m in targets.items():
        name = m.get("label") if m else "NOT FOUND"
        print(f"  {k}: {name}")

    all_rows: list[dict] = []
    for g in ["gemma_base", "gemma_finetuned", "llama_base", "llama_finetuned"]:
        m = targets.get(g)
        print(f"\n--- {g} ---")
        if not m:
            all_rows.append({
                "group": g, "case": "switch", "ok": False,
                "error": "模型未在 Ollama 安装",
            })
            print("  SKIP: not found")
            continue
        rows = run_model_group(g, m)
        all_rows.extend(rows)
        for r in rows:
            if r.get("case") == "switch":
                print(f"  SKIP: {r.get('error')}")
            else:
                flag = "OK" if r.get("ok") else "ISSUE"
                print(f"  [{flag}] {r.get('case'):12} {r.get('elapsed', '-')}s rag={r.get('rag_citations', 0)}")

    summary = summarize(all_rows)
    payload = {
        "timestamp": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z"),
        "targets": {k: (v if v else None) for k, v in targets.items()},
        "summary": summary,
        "rows": all_rows,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(render_md(payload))

    total = len([r for r in all_rows if r.get("case") != "switch"])
    passed = sum(1 for r in all_rows if r.get("ok") and r.get("case") != "switch")
    print(f"\n=== Done: {passed}/{total} passed ===")
    print(f"Report: {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
