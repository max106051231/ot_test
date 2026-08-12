"""
Human Review Queue — Guardrails Reviewer / 分析師覆核管線。

觸發來源：automated status=review|fail、護欄 borderline、LLM 低信心。
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from code.paths import data_dir

DATA_DIR = data_dir()
QUEUE_PATH = DATA_DIR / "review_queue.jsonl"

_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def enqueue_review(
    *,
    item_type: str,
    title: str,
    summary: str,
    control_key: str = "",
    annex_a_id: str = "",
    evidence_id: str = "",
    source: str = "system",
    priority: str = "normal",
    metadata: dict | None = None,
) -> dict:
    item = {
        "review_id": f"RV-{uuid.uuid4().hex[:12]}",
        "item_type": item_type,
        "title": title,
        "summary": summary[:2000],
        "control_key": control_key,
        "annex_a_id": annex_a_id,
        "evidence_id": evidence_id,
        "source": source,
        "priority": priority,
        "status": "pending",
        "reviewer": "",
        "reviewer_notes": "",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "metadata": metadata or {},
    }
    with _lock:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(QUEUE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return item


def list_reviews(status: str | None = None, limit: int = 50) -> list[dict]:
    if not QUEUE_PATH.is_file():
        return []
    rows: list[dict] = []
    with open(QUEUE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if status and rec.get("status") != status:
                continue
            rows.append(rec)
    return rows[-limit:]


def _rewrite_queue(updater) -> bool:
    if not QUEUE_PATH.is_file():
        return False
    rows: list[dict] = []
    with open(QUEUE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    new_rows = updater(rows)
    with open(QUEUE_PATH, "w", encoding="utf-8") as f:
        for rec in new_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return True


def resolve_review(
    review_id: str,
    *,
    status: str,
    reviewer: str = "analyst",
    notes: str = "",
) -> dict | None:
    if status not in ("approved", "rejected"):
        raise ValueError("status must be approved or rejected")
    found: dict | None = None

    def _upd(rows):
        nonlocal found
        out = []
        for rec in rows:
            if rec.get("review_id") == review_id:
                rec = dict(rec)
                rec["status"] = status
                rec["reviewer"] = reviewer
                rec["reviewer_notes"] = notes
                rec["updated_at"] = _now_iso()
                found = rec
            out.append(rec)
        return out

    with _lock:
        _rewrite_queue(_upd)
    return found


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def review_stats() -> dict:
    rows = list_reviews(limit=10000)
    approved = sum(1 for r in rows if r.get("status") == "approved")
    rejected = sum(1 for r in rows if r.get("status") == "rejected")
    pending = sum(1 for r in rows if r.get("status") == "pending")
    decided = approved + rejected

    by_type: dict[str, dict[str, int]] = {}
    for row in rows:
        item_type = (row.get("item_type") or "unknown").strip()
        bucket = by_type.setdefault(
            item_type,
            {"approved": 0, "rejected": 0, "pending": 0, "total": 0},
        )
        bucket["total"] += 1
        status = row.get("status") or "pending"
        if status in bucket:
            bucket[status] += 1

    # 合規審核採納率：不含 guardrail_block（惡意請求本來就應駁回）
    operational_types = ("control_status", "llm_diagnosis")
    op_approved = sum(by_type.get(t, {}).get("approved", 0) for t in operational_types)
    op_rejected = sum(by_type.get(t, {}).get("rejected", 0) for t in operational_types)
    op_decided = op_approved + op_rejected

    cs = by_type.get("control_status") or {}
    cs_approved = cs.get("approved", 0)
    cs_rejected = cs.get("rejected", 0)
    cs_decided = cs_approved + cs_rejected

    return {
        "total": len(rows),
        "pending": pending,
        "approved": approved,
        "rejected": rejected,
        "adoption_rate": _rate(op_approved, op_decided) or _rate(approved, decided),
        "control_mapping_accuracy": _rate(cs_approved, cs_decided),
        "target_adoption_rate": 0.75,
        "target_control_mapping_accuracy": 0.85,
        "by_type": by_type,
        "operational": {
            "approved": op_approved,
            "rejected": op_rejected,
            "decided": op_decided,
            "adoption_rate": _rate(op_approved, op_decided),
        },
    }
