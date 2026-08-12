"""
Evidence 註冊與 hash chain 追溯（No Evidence, No Compliance Claim）。

evidence_id 格式：EV-{site}-{control_key}-{collected_at_utc}-{integrity_hash8}
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from code.paths import data_dir

DATA_DIR = data_dir()
REGISTRY_PATH = DATA_DIR / "evidence_registry.jsonl"
GENESIS_HASH = "0" * 64

_lock = threading.Lock()
_last_hash = GENESIS_HASH


def _site_code() -> str:
    return (os.environ.get("OT_SITE_CODE") or "ot-fab").strip() or "ot-fab"


def _canonical_payload(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _payload_hash(payload: dict) -> str:
    return hashlib.sha256(_canonical_payload(payload).encode("utf-8")).hexdigest()


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def generate_evidence_id(control_key: str, payload: dict, site: str | None = None) -> str:
    site = site or _site_code()
    ts = _utc_compact()
    h8 = _payload_hash(payload)[:8]
    safe_key = (control_key or "general").replace(" ", "_")[:32]
    return f"EV-{site}-{safe_key}-{ts}-{h8}"


def _load_last_hash() -> str:
    global _last_hash
    if not REGISTRY_PATH.is_file():
        return GENESIS_HASH
    last_line = ""
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line
    if not last_line:
        return GENESIS_HASH
    try:
        rec = json.loads(last_line)
        _last_hash = rec.get("payload_hash") or rec.get("integrity_hash") or GENESIS_HASH
        return _last_hash
    except Exception:
        return GENESIS_HASH


def register_evidence(
    *,
    annex_a_id: str,
    control_key: str,
    source_type: str,
    source_file: str = "",
    payload: dict | None = None,
    log_line_ref: str = "",
    device_id: str = "",
    event_count: int = 0,
    metric_summary: str = "",
    collector: str = "semi-shield-collector",
) -> dict:
    """建立 evidence 紀錄並 append 至 registry（含 hash chain）。"""
    global _last_hash
    payload = dict(payload or {})
    payload.update({
        "annex_a_id": annex_a_id,
        "control_key": control_key,
        "source_type": source_type,
        "source_file": source_file,
        "log_line_ref": log_line_ref,
        "device_id": device_id,
        "event_count": event_count,
        "metric_summary": metric_summary,
    })
    collected_at = datetime.now(timezone.utc).isoformat()
    integrity = _payload_hash(payload)
    evidence_id = generate_evidence_id(control_key, payload)

    with _lock:
        prev_hash = _load_last_hash()
        record = {
            "evidence_id": evidence_id,
            "annex_a_id": annex_a_id,
            "control_key": control_key,
            "source_type": source_type,
            "source_file": source_file,
            "log_line_ref": log_line_ref,
            "device_id": device_id,
            "event_count": event_count,
            "metric_summary": metric_summary,
            "collected_at": collected_at,
            "collector": collector,
            "payload": payload,
            "payload_hash": integrity,
            "integrity_hash": integrity[:8],
            "prev_hash": prev_hash,
            "review_status": "pending",
        }
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(REGISTRY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        _last_hash = integrity
    return record


def register_control_bundle_evidence(
    control_key: str,
    bundle: dict,
    *,
    annex_a_id: str = "",
    source_file: str = "",
) -> dict:
    """為 OT 控制項摘要 bundle 建立 aggregate evidence。"""
    if not annex_a_id:
        try:
            from code.services.compliance_service import control_by_key

            meta = control_by_key(control_key) or {}
            annex_a_id = meta.get("annex_a_id") or control_key
        except Exception:
            annex_a_id = control_key
    return register_evidence(
        annex_a_id=annex_a_id,
        control_key=control_key,
        source_type="syslog_aggregate",
        source_file=source_file,
        event_count=int(bundle.get("event_count") or 0),
        metric_summary=str(bundle.get("metric_summary") or ""),
        payload={
            "title": bundle.get("title"),
            "log_excerpt": (bundle.get("log") or "")[:500],
        },
    )


def list_evidence(limit: int = 100, control_key: str | None = None) -> list[dict]:
    if not REGISTRY_PATH.is_file():
        return []
    rows: list[dict] = []
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if control_key and rec.get("control_key") != control_key:
                continue
            rows.append(rec)
    return rows[-limit:]


def get_evidence(evidence_id: str) -> dict | None:
    if not REGISTRY_PATH.is_file():
        return None
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("evidence_id") == evidence_id:
                return rec
    return None


def update_evidence_review_status(
    evidence_id: str,
    *,
    review_status: str,
    reviewer: str = "ai-reviewer",
    review_note: str = "",
) -> dict | None:
    """更新 evidence 審核狀態（approved / rejected）。"""
    if review_status not in ("approved", "rejected", "pending"):
        raise ValueError("review_status must be approved, rejected, or pending")
    found: dict | None = None

    def _upd(rows):
        nonlocal found
        out = []
        for rec in rows:
            if rec.get("evidence_id") == evidence_id:
                rec = dict(rec)
                rec["review_status"] = review_status
                rec["reviewer"] = reviewer
                rec["human_review_note"] = review_note
                rec["reviewed_at"] = datetime.now(timezone.utc).isoformat()
                found = rec
            out.append(rec)
        return out

    with _lock:
        if not REGISTRY_PATH.is_file():
            return None
        rows: list[dict] = []
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
        new_rows = _upd(rows)
        if not found:
            return None
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            for rec in new_rows:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return found


def traceability_stats() -> dict:
    rows = list_evidence(limit=10000)
    total = len(rows)
    with_id = sum(1 for r in rows if r.get("evidence_id"))
    with_hash = sum(1 for r in rows if r.get("payload_hash"))
    pending = sum(1 for r in rows if r.get("review_status") == "pending")
    rate = round(with_id / total, 4) if total else 0.0
    hash_rate = round(with_hash / total, 4) if total else 0.0
    return {
        "total_evidence_records": total,
        "with_evidence_id": with_id,
        "with_payload_hash": with_hash,
        "traceability_rate": max(rate, hash_rate),
        "pending_review": pending,
    }
