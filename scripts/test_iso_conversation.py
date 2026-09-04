#!/usr/bin/env python3
"""
多輪 ISO27001 合規對話測試（模擬 agent_chat.html 的 history 流程）。
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "http://127.0.0.1:2000"
ROOT = Path(__file__).resolve().parent.parent

QUESTIONS = [
    ("q1_iso_compliant", "我們現在符合 ISO27001 嗎？"),
    ("q2_why_uncertain", "為什麼你不能確定？"),
    ("q3_evidence_gap", "哪些 evidence 不足？"),
    ("q4_coverage_device", "哪一台設備造成 monitoring coverage 下降？"),
    ("q5_chart", "給我圖表。"),
    ("q6_controls", "這會影響哪些控制項？"),
    ("q7_next_steps", "下一步要做什麼？"),
]

BAD_PATTERNS = [
    (r"請再問一次|模型輸出不穩|已攔截|無法根據目前提供", "bad_fallback"),
    (r"Ollama 推論失敗|Ollama 未連線|模型未成功載入", "llm_error"),
    (r"^Please provide", "english_only"),
    (r"分析使用者|使用者本則問題", "meta_leak"),
]


def post(path: str, body: dict, timeout: float = 300) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get(path: str, timeout: float = 30) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def switch_model(slug: str) -> tuple[bool, str]:
    try:
        r = post("/api/llm/switch", {"model": slug})
        if not r.get("ok"):
            return False, r.get("error") or "switch failed"
        return True, r.get("label") or slug
    except Exception as e:
        return False, str(e)


def analyze(qid: str, reply: str, data: dict) -> list[str]:
    issues: list[str] = []
    text = (reply or "").strip()
    status = data.get("status")

    if status == "blocked":
        issues.append("guardrail_blocked")
    elif status != "success":
        issues.append(f"status_{status}")

    if not text:
        issues.append("empty_reply")
    elif len(text) < 12:
        issues.append("too_short")

    for pat, tag in BAD_PATTERNS:
        if re.search(pat, text, re.I | re.M):
            issues.append(tag)

    if text and not re.search(r"[\u4e00-\u9fff]", text) and len(text) > 30:
        issues.append("no_chinese")

    if qid == "q5_chart":
        if "```chart" not in text and "chart" not in text.lower():
            # 後端可能補 chart；至少要有實質說明
            if len(text) < 40:
                issues.append("chart_weak")

    if qid in ("q1_iso_compliant", "q3_evidence_gap", "q6_controls"):
        if text and not re.search(r"27001|ISO|控制|合規|evidence|Annex|A\.\d", text, re.I):
            issues.append("off_topic")

    gr = data.get("guardrail") or {}
    if gr.get("blocked"):
        issues.append("guardrail_blocked")

    return issues


def run_conversation(model_slug: str, model_label: str) -> dict:
    history: list[dict] = []
    turns: list[dict] = []
    ok_count = 0

    for qid, message in QUESTIONS:
        t0 = time.time()
        body = {"message": message, "history": history[-6:]}
        if model_slug:
            body["model"] = model_slug
        try:
            data = post("/api/agent/chat", body)
            elapsed = round(time.time() - t0, 1)
        except Exception as e:
            turns.append({
                "id": qid,
                "question": message,
                "status": "error",
                "error": str(e),
                "issues": ["request_error"],
            })
            continue

        reply = data.get("reply") or ""
        issues = analyze(qid, reply, data)
        if not issues:
            ok_count += 1

        turns.append({
            "id": qid,
            "question": message,
            "status": data.get("status"),
            "elapsed_s": elapsed,
            "issues": issues,
            "tool_called": data.get("tool_called"),
            "tool_name": data.get("tool_name"),
            "rag_citations": len(data.get("rag_citations") or []),
            "reply_preview": re.sub(r"\s+", " ", reply)[:200],
            "reply_len": len(reply),
        })

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})

    return {
        "model_slug": model_slug,
        "model_label": model_label,
        "passed": ok_count,
        "total": len(QUESTIONS),
        "turns": turns,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=== ISO27001 Multi-turn Conversation Test ===\n")
    try:
        safety = get("/api/safety/status")
    except Exception as e:
        print(f"FAIL: server not reachable at {BASE}: {e}")
        return 1

    models = [
        m for m in (safety.get("llm_models") or [])
        if m.get("available")
    ]
    if not models:
        print("FAIL: no available models")
        return 1

    print(f"Server OK | models={len(models)} RAG={safety.get('rag_enabled')}")
    print("Questions:", [q[1] for q in QUESTIONS], "\n")

    all_results: list[dict] = []
    total_fail = 0

    for m in models:
        slug = m.get("slug") or ""
        label = m.get("label") or slug
        stage = m.get("stage") or ""
        print(f"--- {label} ({slug}) [{stage}] ---")
        ok, msg = switch_model(slug)
        if not ok:
            print(f"  SKIP switch failed: {msg}\n")
            all_results.append({
                "model_slug": slug,
                "model_label": label,
                "passed": 0,
                "total": len(QUESTIONS),
                "skip": msg,
                "turns": [],
            })
            total_fail += len(QUESTIONS)
            continue

        result = run_conversation(slug, msg)
        all_results.append(result)
        for t in result["turns"]:
            flag = "OK" if not t.get("issues") else "FAIL"
            if t.get("issues"):
                total_fail += 1
            prev = t.get("reply_preview", "")[:100]
            print(
                f"  [{flag}] {t['id']:18} {t.get('elapsed_s', 0):5.1f}s "
                f"rag={t.get('rag_citations', 0)} tool={t.get('tool_name') or '-'}"
            )
            print(f"         {prev}")
            if t.get("issues"):
                print(f"         issues: {', '.join(t['issues'])}")
        print(
            f"  => {result['passed']}/{result['total']} passed\n"
        )

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": BASE,
        "questions": QUESTIONS,
        "results": all_results,
        "summary": {
            "models_tested": len(all_results),
            "total_turns": len(all_results) * len(QUESTIONS),
            "failed_turns": total_fail,
        },
    }
    out_path = ROOT / "scripts" / "test_iso_conversation_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = ROOT / "docs" / "ISO_CONVERSATION_TEST_REPORT.md"
    lines = [
        "# ISO27001 多輪對話測試報告",
        "",
        f"- 時間：{out['generated_at']}",
        f"- 模型數：{len(all_results)}",
        f"- 失敗回合：{total_fail} / {len(all_results) * len(QUESTIONS)}",
        "",
        "## 問題序列",
        "",
    ]
    for i, (_, q) in enumerate(QUESTIONS, 1):
        lines.append(f"{i}. {q}")
    lines.append("")
    for r in all_results:
        lines.append(f"## {r.get('model_label')} (`{r.get('model_slug')}`)")
        lines.append("")
        if r.get("skip"):
            lines.append(f"SKIP: {r['skip']}")
            lines.append("")
            continue
        lines.append(f"通過：**{r['passed']}/{r['total']}**")
        lines.append("")
        lines.append("| # | 問題 | 結果 | 秒 | RAG | 問題 |")
        lines.append("|---|------|------|-----|-----|------|")
        for t in r.get("turns", []):
            iss = ", ".join(t.get("issues") or []) or "—"
            lines.append(
                f"| {t['id']} | {t['question'][:24]} | {t.get('status')} | "
                f"{t.get('elapsed_s', '-')} | {t.get('rag_citations', 0)} | {iss} |"
            )
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"JSON: {out_path}")
    print(f"Report: {md_path}")
    print(f"\nSUMMARY: failed_turns={total_fail}")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
