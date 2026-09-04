#!/usr/bin/env python3
"""Quick 7-turn test for selected models (unbuffered stdout)."""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:2000"
QUESTIONS = [
    "我們現在符合 ISO27001 嗎？",
    "為什麼你不能確定？",
    "哪些 evidence 不足？",
    "哪一台設備造成 monitoring coverage 下降？",
    "給我圖表。",
    "這會影響哪些控制項？",
    "下一步要做什麼？",
]

# 先測微調後代表模型；全量測試用 test_iso_conversation.py
MODELS = sys.argv[1:] or [
    "gemma_2b_ot",
    "llama32_3b_ot",
    "qwen25_3b_ot",
    "phi4_mini_ot",
]


def post(path, body, timeout=300):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    for slug in MODELS:
        r = post("/api/llm/switch", {"model": slug})
        ok = r.get("ok")
        label = r.get("label") or slug
        print(f"\n=== {label} ({slug}) switch={ok} ===", flush=True)
        if not ok:
            print(f"  skip: {r.get('error')}", flush=True)
            continue
        hist = []
        passed = 0
        for i, q in enumerate(QUESTIONS, 1):
            t0 = time.time()
            try:
                d = post("/api/agent/chat", {"message": q, "history": hist[-6:]})
                el = round(time.time() - t0, 1)
                rep = (d.get("reply") or "").replace("\n", " ")
                prev = rep[:140]
                issues = []
                if d.get("status") != "success":
                    issues.append(f"status={d.get('status')}")
                if (d.get("guardrail") or {}).get("blocked"):
                    issues.append("blocked")
                if not rep.strip():
                    issues.append("empty")
                if re.search(r"請再問一次|已攔截|無法根據", rep):
                    issues.append("fallback")
                flag = "OK" if not issues else "FAIL"
                if not issues:
                    passed += 1
                rag_n = len(d.get("rag_citations") or [])
                tool = d.get("tool_name") or "-"
                print(
                    f"[{flag}] Q{i} {el}s rag={rag_n} tool={tool} | {prev}",
                    flush=True,
                )
                if issues:
                    print(f"      issues: {', '.join(issues)}", flush=True)
                hist += [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": d.get("reply") or ""},
                ]
            except Exception as e:
                print(f"[ERR] Q{i} {e}", flush=True)
        print(f"=> {passed}/{len(QUESTIONS)} passed", flush=True)


if __name__ == "__main__":
    main()
