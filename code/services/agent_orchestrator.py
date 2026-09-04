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
        "tools": ["build_control_log_bundle", "export_pdf"],
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
            "2. Reviewer: enqueue review/fail 控制項至 Human Review Queue",
            "3. Reporter: 彙整 metrics / evidence（PDF 至監控戰情室匯出）",
            "— 另按鈕：AI 自動審核（POST /api/review/auto）處理 pending queue",
            "— 可選：LLM 診斷（POST /api/compliance/pipeline {\"include_llm_audit\":true} 或 /api/audit/analyze_all）",
        ],
        "design_principle": "No Evidence, No Compliance Claim",
    }


def run_compliance_pipeline(
    *,
    scan_fn: Callable[[], dict | None],
    audit_fn: Callable[[str, dict], dict] | None = None,
    enqueue_fn: Callable[..., dict],
    evidence_fn: Callable[[str, dict], dict],
    include_llm_audit: bool = False,
) -> dict[str, Any]:
    """
    執行地端合規管線（預設不含 LLM 診斷／AI 自動審核）。
    - 預設：Collector → Reviewer（入列）→ Reporter
    - include_llm_audit=True：額外執行 Auditor（RAG + ask_llm）
    AI Reviewer（/api/review/auto）須由前端另按「AI 自動審核」觸發。
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
    enqueued = 0

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
            enqueued += 1
        if include_llm_audit and audit_fn and (
            status != "pass" or int(bundle.get("event_count") or 0) > 0
        ):
            audit_results[key] = audit_fn(key, bundle)

    if include_llm_audit and audit_fn:
        stages.append({
            "stage": "Auditor",
            "status": "ok",
            "diagnosed": list(audit_results.keys()),
        })
    else:
        stages.append({
            "stage": "Auditor",
            "status": "skipped",
            "hint": "未執行 LLM 診斷；請至監控戰情室或 POST /api/audit/analyze_all",
        })

    stages.append({
        "stage": "Reviewer",
        "status": "ok",
        "evidence_ids": evidence_map,
        "enqueued": enqueued,
    })
    stages.append({
        "stage": "Reporter",
        "status": "ready",
        "control_count": len(bundles),
        "evidence_map": evidence_map,
        "report_export": "/monitor",
        "report_hint": "請至監控戰情室匯出稽核報告 PDF（含 evidence 附錄）",
        "audit_results": (
            {k: v.get("ai_analysis", "")[:200] for k, v in audit_results.items()}
            if audit_results
            else {}
        ),
    })

    return {
        "ok": True,
        "stages": stages,
        "evidence_map": evidence_map,
        "metrics": metrics,
        "include_llm_audit": include_llm_audit,
        "llm_audit_count": len(audit_results),
    }
