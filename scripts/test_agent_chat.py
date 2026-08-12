#!/usr/bin/env python3
"""Smoke-test /api/agent/chat across common LLM reply paths."""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:2000"

CASES = [
    ("casual", "你好"),
    ("patch_10", "給我10個修補建議"),
    ("compliance", "目前合規現況如何？"),
    ("iso_knowledge", "ISO 27001 Annex A 是什麼？"),
    ("hardening", "如何停用 Telnet 並改用 SSH？"),
    ("chart", "用圓餅圖看各控制項事件占比"),
    ("off_topic", "今天台東天氣如何？"),
    ("follow_up", "剛才說的 A.5.15 要怎麼改善？"),
]

MODELS = [
    "",
    "qwen2.5:3b",
    "gemma_2b_ot",
    "qwen25_3b_ot",
]

BAD_PATTERNS = [
    (r"Ollama 推論失敗|模型輸出異常|輸出異常", "llm_error"),
    (r"\[UNK_BYTE_", "gemma_corrupt"),
    (r"(?:</>\s*){3,}", "gemma_tag_spam"),
    (r"peg-native format|Ollama HTTP 500", "ollama_crash"),
    (r"請再問一次，或改問", "generic_fallback"),
    (r"^Please provide me with some context", "english_only_gemma"),
    (r"分析使用者|使用者本則問題|可能的回應", "meta_leak"),
    (r"⚠️.*Ollama 未連線", "ollama_down"),
]


def post(path: str, body: dict, timeout: float = 180) -> dict:
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


def analyze_reply(reply: str) -> list[str]:
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
    return issues


def switch_model(model: str) -> tuple[bool, str]:
    if not model:
        return True, "default"
    try:
        r = post("/api/llm/switch", {"model": model})
        if not r.get("ok"):
            return False, r.get("error") or "switch failed"
        return True, r.get("label") or model
    except Exception as e:
        return False, str(e)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=== Agent Chat LLM Smoke Test ===\n")
    try:
        safety = get("/api/safety/status")
    except Exception as e:
        print(f"FAIL: cannot reach server at {BASE}: {e}")
        return 1

    llm = safety.get("llm_model") or {}
    print(f"Server OK | backend={llm.get('backend')} current={llm.get('label') or llm.get('slug')}")
    print(f"RAG={safety.get('rag', {}).get('enabled')} Guardrail={safety.get('guardrail', {}).get('enabled')}\n")

    results: list[dict] = []
    total_fail = 0

    for model in MODELS:
        label = model or "(keep current)"
        ok, msg = switch_model(model)
        if not ok:
            print(f"SKIP model {label}: {msg}\n")
            for case_id, _ in CASES:
                results.append({
                    "model": label,
                    "case": case_id,
                    "status": "skip_switch",
                    "issues": [msg],
                })
                total_fail += 1
            continue

        print(f"--- Model: {msg} ---")
        history: list[dict] = []

        for case_id, message in CASES:
            t0 = time.time()
            try:
                body = {"message": message, "history": history[-6:]}
                if model:
                    body["model"] = model
                data = post("/api/agent/chat", body)
                elapsed = round(time.time() - t0, 1)
                status = data.get("status", "?")
                reply = data.get("reply") or ""
                issues = analyze_reply(reply) if status == "success" else [f"http_status_{status}"]
                if status == "blocked":
                    issues = ["guardrail_blocked"]

                # patch_10: expect numbered list
                if case_id == "patch_10" and status == "success":
                    nums = re.findall(r"(?m)^\s*(\d+)\.\s+", reply)
                    if len(nums) < 5:
                        issues.append("patch_not_numbered")

                preview = re.sub(r"\s+", " ", reply)[:120]
                flag = "OK" if not issues else "ISSUE"
                if issues:
                    total_fail += 1
                print(f"  [{flag}] {case_id:14} {elapsed:5.1f}s | {preview}")
                if issues:
                    print(f"         issues: {', '.join(issues)}")

                results.append({
                    "model": msg,
                    "case": case_id,
                    "status": status,
                    "elapsed": elapsed,
                    "issues": issues,
                    "reply_len": len(reply),
                    "tool": data.get("tool_name"),
                })

                if status == "success" and reply:
                    history.append({"role": "user", "content": message})
                    history.append({"role": "assistant", "content": reply[:800]})

            except urllib.error.HTTPError as e:
                total_fail += 1
                err_body = e.read().decode("utf-8", errors="replace")[:200]
                print(f"  [FAIL] {case_id:14} HTTP {e.code} | {err_body}")
                results.append({
                    "model": msg,
                    "case": case_id,
                    "status": f"http_{e.code}",
                    "issues": [err_body],
                })
            except Exception as e:
                total_fail += 1
                print(f"  [FAIL] {case_id:14} {e}")
                results.append({
                    "model": msg,
                    "case": case_id,
                    "status": "error",
                    "issues": [str(e)],
                })
        print()

    out_path = "scripts/test_agent_chat_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    passed = sum(1 for r in results if not r.get("issues"))
    print(f"Summary: {passed}/{len(results)} passed, {total_fail} with issues")
    print(f"Details: {out_path}")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
