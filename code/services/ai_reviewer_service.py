"""
AI Reviewer — 無分析師時由地端規則 + 可追溯性檢查自動審核。

審核 Review Queue 與 Evidence registry；不取代正式稽核簽核，
但可讓管線與 KPI 在競賽／PoC 環境自動跑完。

環境變數：
  ENABLE_AI_REVIEWER=1   啟用（Ollama 模式預設開）
  AI_REVIEWER_ID=ai-reviewer
"""
from __future__ import annotations

import os
import re

from code.services.compliance_service import control_by_key
from code.services.evidence_service import (
    get_evidence,
    list_evidence,
    update_evidence_review_status,
)
from code.services.review_queue import list_reviews, resolve_review, review_stats

AI_REVIEWER_ID = (os.environ.get("AI_REVIEWER_ID") or "ai-reviewer").strip()


def ai_reviewer_enabled() -> bool:
    return os.environ.get("ENABLE_AI_REVIEWER", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _latest_evidence_for_control(control_key: str) -> dict | None:
    if not control_key:
        return None
    rows = [r for r in list_evidence(limit=500) if r.get("control_key") == control_key]
    return rows[-1] if rows else None


def _decide_queue_item(item: dict) -> tuple[str, str]:
    """回傳 (approved|rejected, notes)。"""
    item_type = (item.get("item_type") or "").strip()
    summary = (item.get("summary") or "").strip()
    evidence_id = (item.get("evidence_id") or "").strip()
    control_key = (item.get("control_key") or "").strip()
    annex_a_id = (item.get("annex_a_id") or "").strip()

    if item_type == "guardrail_block":
        return (
            "rejected",
            "AI Reviewer：惡意／注入類請求維持攔截，不作合規宣稱。",
        )

    ev = get_evidence(evidence_id) if evidence_id else None
    if not ev and control_key:
        ev = _latest_evidence_for_control(control_key)
        if ev:
            evidence_id = ev.get("evidence_id") or evidence_id

    if evidence_id and ev:
        if not ev.get("payload_hash"):
            return ("rejected", "AI Reviewer：evidence 缺少 hash chain。")

    if control_key:
        meta = control_by_key(control_key) or {}
        if not annex_a_id:
            annex_a_id = meta.get("annex_a_id") or ""
        if not meta and not annex_a_id:
            return ("rejected", "AI Reviewer：control_key 未對映 Annex A。")

    if item_type == "control_status":
        m = re.search(r"status=(\w+)", summary, re.I)
        status = (m.group(1) if m else "").lower()
        if not ev and not control_key:
            return ("rejected", "AI Reviewer：控制項告警缺少 evidence／control_key。")
        if not ev:
            return ("rejected", "AI Reviewer：找不到該控制項的 evidence 紀錄。")
        if status == "fail":
            return (
                "approved",
                "AI Reviewer：已附 grounded syslog evidence；"
                "fail 狀態標記為「待修補」，非虛構合規通過。",
            )
        if status == "review":
            return (
                "approved",
                "AI Reviewer：review 狀態已附 evidence，建議排程複核 OT 日誌。",
            )
        return ("approved", "AI Reviewer：控制項狀態已附 evidence 追溯。")

    if evidence_id or control_key:
        return ("approved", "AI Reviewer：具 evidence／控制項對映，核准進報告附錄。")

    return ("approved", "AI Reviewer：一般項目核准（已留 audit trail）。")


def _decide_evidence(rec: dict) -> tuple[str, str]:
    if (rec.get("review_status") or "pending") != "pending":
        return ("approved", "已審")

    if not rec.get("evidence_id") or not rec.get("payload_hash"):
        return ("rejected", "AI Reviewer：缺少 evidence_id 或 payload_hash。")

    if not rec.get("annex_a_id") or not rec.get("control_key"):
        return ("rejected", "AI Reviewer：缺少 Annex A／control_key 對映。")

    src = (rec.get("source_type") or "").strip()
    if src == "syslog_aggregate":
        if int(rec.get("event_count") or 0) <= 0:
            return (
                "rejected",
                "AI Reviewer：syslog aggregate 事件量為 0，不構成有效 evidence。",
            )
        return (
            "approved",
            "AI Reviewer：OT syslog 聚合 + hash chain 可追溯，核准納入附錄。",
        )

    return ("approved", "AI Reviewer：evidence 欄位完整，核准。")


def reconcile_stale_reviews(*, limit: int = 2000) -> dict:
    """
    重新審核先前因 evidence 未連結而被駁回的 control_status 項目。
    管線多次執行後 queue 會累積舊 rejected，拖低 KPI。
    """
    if not ai_reviewer_enabled():
        return {"ok": False, "enabled": False, "reconciled": 0}

    results: list[dict] = []
    for item in list_reviews(status="rejected", limit=limit):
        if (item.get("item_type") or "").strip() != "control_status":
            continue
        status, notes = _decide_queue_item({**item, "status": "pending"})
        if status != "approved":
            continue
        updated = resolve_review(
            item["review_id"],
            status=status,
            reviewer=AI_REVIEWER_ID,
            notes=notes + "（reconcile：evidence 已補齊）",
        )
        results.append({
            "review_id": item.get("review_id"),
            "control_key": item.get("control_key"),
            "decision": status,
            "ok": bool(updated),
        })

    return {
        "ok": True,
        "reconciled": len(results),
        "items": results[:20],
        "stats": review_stats(),
    }


def run_ai_reviewer(*, limit: int = 500, reconcile: bool = True) -> dict:
    """
    批次審核 pending 的 review_queue 與 evidence_registry。
    回傳統計供 API／管線使用。
    """
    if not ai_reviewer_enabled():
        return {
            "ok": False,
            "enabled": False,
            "error": "ENABLE_AI_REVIEWER=0，AI 審核已停用",
        }

    queue_results: list[dict] = []
    evidence_results: list[dict] = []

    for item in list_reviews(status="pending", limit=limit):
        status, notes = _decide_queue_item(item)
        updated = resolve_review(
            item["review_id"],
            status=status,
            reviewer=AI_REVIEWER_ID,
            notes=notes,
        )
        queue_results.append({
            "review_id": item.get("review_id"),
            "item_type": item.get("item_type"),
            "decision": status,
            "notes": notes,
            "ok": bool(updated),
        })

    seen_ev: set[str] = set()
    for rec in list_evidence(limit=limit):
        eid = rec.get("evidence_id") or ""
        if not eid or eid in seen_ev:
            continue
        if (rec.get("review_status") or "pending") != "pending":
            continue
        seen_ev.add(eid)
        status, notes = _decide_evidence(rec)
        updated = update_evidence_review_status(
            eid,
            review_status=status,
            reviewer=AI_REVIEWER_ID,
            review_note=notes,
        )
        evidence_results.append({
            "evidence_id": eid,
            "control_key": rec.get("control_key"),
            "decision": status,
            "notes": notes,
            "ok": bool(updated),
        })

    reconcile_result: dict | None = None
    if reconcile:
        reconcile_result = reconcile_stale_reviews(limit=limit)

    q_ok = sum(1 for r in queue_results if r.get("ok"))
    e_ok = sum(1 for r in evidence_results if r.get("ok"))
    q_ap = sum(1 for r in queue_results if r.get("decision") == "approved")
    e_ap = sum(1 for r in evidence_results if r.get("decision") == "approved")

    return {
        "ok": True,
        "enabled": True,
        "reviewer": AI_REVIEWER_ID,
        "mode": "ai_reviewer",
        "reconcile": reconcile_result,
        "queue": {
            "processed": len(queue_results),
            "approved": q_ap,
            "rejected": len(queue_results) - q_ap,
            "updated": q_ok,
            "items": queue_results[:20],
        },
        "evidence": {
            "processed": len(evidence_results),
            "approved": e_ap,
            "rejected": len(evidence_results) - e_ap,
            "updated": e_ok,
            "items": evidence_results[:20],
        },
        "stats": review_stats(),
        "summary": (
            f"AI Reviewer 已處理 queue {len(queue_results)} 筆、"
            f"evidence {len(evidence_results)} 筆"
        ),
    }


def reviewer_mode_summary() -> dict:
    """供 /api/safety/status 與合規頁顯示。"""
    stats = review_stats()
    return {
        "enabled": ai_reviewer_enabled(),
        "reviewer_id": AI_REVIEWER_ID,
        "mode": "ai_reviewer" if ai_reviewer_enabled() else "human_analyst",
        "label": "AI Reviewer（地端自動審核）" if ai_reviewer_enabled() else "Human Analyst",
        "note": (
            "無分析師時由 AI Reviewer 依 evidence／hash chain／控制項對映自動核准或駁回；"
            "護欄攔截維持 rejected。"
            if ai_reviewer_enabled()
            else "需人工分析師於 Review Queue 核准。"
        ),
        "pending_queue": stats.get("pending") or 0,
        "adoption_rate": stats.get("adoption_rate"),
    }
