"""
Agent Workflow 編排（Collector → Auditor → Reviewer → Reporter）。

現行為輕量順序管線；Phase 2 可換 LangGraph / CrewAI。
通訊：同程序 Python 呼叫（無外部 message bus）。
"""
from __future__ import annotations

from typing import Any, Callable

AGENT_ROLES = [
    {
        "role": "Collector",
        "agent_id": "agent-collector",
        "duty": "掃描 OT 目錄 / Wazuh 事件，註冊 syslog evidence",
        "tools": ["scan_ot_directory", "evidence_service.register_evidence"],
    },
    {
        "role": "Auditor",
        "agent_id": "agent-auditor",
        "duty": "RAG 檢索 + Local LLM 控制項診斷",
        "tools": ["retrieve_rag_for_query", "ask_llm"],
    },
    {
        "role": "Reviewer",
        "agent_id": "agent-reviewer",
        "duty": "Guardrails 邊界判定 + Human review queue",
        "tools": ["guardrail_service.check_input", "review_queue.enqueue_review"],
    },
    {
        "role": "Reporter",
        "agent_id": "agent-reporter",
        "duty": "彙整 metrics / evidence / AI 分析 → 合規報告",
        "tools": ["build_control_log_bundle", "export_pdf_txt"],
    },
]

FRAMEWORK = {
    "current": "lightweight_sequential",
    "recommended_phase2": "LangGraph 或 CrewAI",
    "communication": "in-process function calls",
    "agent_count": len(AGENT_ROLES),
}


def workflow_spec() -> dict:
    return {
        "framework": FRAMEWORK,
        "agents": AGENT_ROLES,
        "pipeline": [
            "1. Collector: scan_ot → evidence_id per control_key",
            "2. Auditor: RAG + LLM diagnosis per control",
            "3. Reviewer: guardrail + enqueue if review/fail/borderline",
            "4. Reporter: bundle PDF/TXT with evidence appendix",
        ],
        "design_principle": "No Evidence, No Compliance Claim",
    }


def run_compliance_pipeline(
    *,
    scan_fn: Callable[[], dict | None],
    audit_fn: Callable[[str, dict], dict],
    enqueue_fn: Callable[..., dict],
    evidence_fn: Callable[[str, dict], dict],
) -> dict[str, Any]:
    """
    執行四階段合規管線。由 app.py 注入實際函式避免循環 import。
    """
    stages: list[dict] = []
    scan_data = scan_fn() or {}
    if scan_data.get("error"):
        return {"ok": False, "error": scan_data["error"], "stages": stages}

    stages.append({"stage": "Collector", "status": "ok", "files": scan_data.get("summary", {})})

    bundles = scan_data.get("control_bundles") or {}
    metrics = scan_data.get("metrics") or {}
    evidence_map: dict[str, str] = {}
    audit_results: dict[str, Any] = {}

    for key, bundle in bundles.items():
        ev = evidence_fn(key, bundle)
        evidence_map[key] = ev.get("evidence_id", "")
        metric = metrics.get(key) or {}
        status = metric.get("status", "pass")
        if status in ("review", "fail"):
            enqueue_fn(
                item_type="control_status",
                title=f"{bundle.get('title', key)} 需覆核",
                summary=f"status={status}; {bundle.get('metric_summary', '')}",
                control_key=key,
                evidence_id=ev.get("evidence_id", ""),
                priority="high" if status == "fail" else "normal",
            )
        if status != "pass" or int(bundle.get("event_count") or 0) > 0:
            audit_results[key] = audit_fn(key, bundle)

    stages.append({
        "stage": "Auditor",
        "status": "ok",
        "diagnosed": list(audit_results.keys()),
    })
    stages.append({
        "stage": "Reviewer",
        "status": "ok",
        "evidence_ids": evidence_map,
    })
    stages.append({
        "stage": "Reporter",
        "status": "ready",
        "control_count": len(bundles),
        "evidence_map": evidence_map,
        "report_export": "/monitor",
        "report_hint": "請至監控戰情室匯出 PDF／TXT（含 evidence 附錄）",
        "audit_results": {k: v.get("ai_analysis", "")[:200] for k, v in audit_results.items()},
    })

    return {
        "ok": True,
        "stages": stages,
        "evidence_map": evidence_map,
        "metrics": metrics,
    }
