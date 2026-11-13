"""監控戰情室：人工稽核資料匯入（政策／GRC／訪談紀錄等）。"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from code.paths import data_dir

_STORE = data_dir() / "manual_imports.jsonl"
_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def list_imports(*, annex_a_id: str | None = None, limit: int = 200) -> list[dict]:
    if not _STORE.is_file():
        return []
    aid = (annex_a_id or "").strip()
    out: list[dict] = []
    with _lock:
        for line in _STORE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if aid and (rec.get("annex_a_id") or "").strip() != aid:
                continue
            out.append(rec)
    out.sort(key=lambda r: r.get("imported_at") or "", reverse=True)
    return out[:limit]


def count_by_annex() -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in list_imports(limit=5000):
        aid = (rec.get("annex_a_id") or "").strip()
        if aid:
            counts[aid] = counts.get(aid, 0) + 1
    return counts


def append_import(
    *,
    annex_a_id: str,
    title: str = "",
    content: str,
    source: str = "manual_upload",
    simulated: bool = False,
    metadata: dict | None = None,
) -> dict:
    aid = (annex_a_id or "").strip()
    body = (content or "").strip()
    if not aid:
        raise ValueError("annex_a_id 必填")
    if not body:
        raise ValueError("content 必填")
    rec = {
        "import_id": f"mi-{uuid.uuid4().hex[:12]}",
        "annex_a_id": aid,
        "title": (title or aid).strip(),
        "content": body[:12000],
        "source": source,
        "simulated": bool(simulated),
        "imported_at": _now_iso(),
        "metadata": metadata or {},
    }
    _STORE.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with open(_STORE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def log_text_for_annex(annex_a_id: str, *, max_records: int = 4) -> str:
    rows = list_imports(annex_a_id=annex_a_id, limit=max_records)
    if not rows:
        return ""
    lines: list[str] = []
    for rec in rows:
        tag = "[SIM]" if rec.get("simulated") else "[MANUAL]"
        lines.append(
            f"{tag} {rec.get('imported_at', '—')} · {rec.get('source', 'manual')} · "
            f"{rec.get('title', annex_a_id)}"
        )
        for part in (rec.get("content") or "").splitlines()[:6]:
            lines.append(part)
        lines.append("---")
    total = len(list_imports(annex_a_id=annex_a_id, limit=5000))
    if total > max_records:
        lines.append(f"...共 {total} 筆人工匯入紀錄，以上顯示最新 {max_records} 筆...")
    return "\n".join(lines).strip()


def _simulated_body_for_control(control: dict) -> str:
    aid = (control.get("annex_a_id") or "").strip()
    title = (control.get("title") or aid).strip()
    domain = control.get("domain") or "organizational"
    etypes = control.get("evidence_types") or ["manual_review"]
    etype_line = " · ".join(str(x) for x in etypes[:6])
    domain_hint = {
        "organizational": "政策／程序／GRC 匯出",
        "people": "HR 訓練紀錄／離職清單／NDA",
        "physical": "門禁／機房巡檢／CCTV 抽樣表",
        "technological": "設定審查／變更紀錄／掃描報告摘要",
    }.get(domain, "稽核訪談／文件化 evidence")
    extras = _SIM_BODY_EXTRAS.get(aid)
    if extras:
        return extras
    return (
        f"[SIMULATED · AI 匯入管線 · {aid}]\n"
        f"控制項：{title}\n"
        f"證據類型：{etype_line}\n"
        f"來源類別：{domain_hint}\n"
        f"涵蓋期間：2026-Q1 · Fab-A/B OT 場域\n"
        f"完整性：必填欄位已勾稽 · 版本 v1 · 示範批次 simulate_manual_batch\n"
        f"摘要：已對照 ISO/IEC 27001:2022 Annex A {aid}，供 Control Bundle 與證據鏈展示。"
    )


# 少數控制項保留較擬真的示範原文
_SIM_BODY_EXTRAS: dict[str, str] = {
    "A.5.1": (
        "[POLICY v3.1]\n"
        "核准：資安長 2026-03-01\n"
        "摘要：全廠 ISMS 適用範圍含 Fab-A/B OT 監控與 syslog 留存；"
        "員工與供應商須遵守 A.5.19 遠端維運規範。\n"
        "附件：isms_policy_A51_signed.pdf"
    ),
    "A.5.2": (
        "[RACI 匯出]\n"
        "CISO：政策與風險核准\n"
        "OT 維運主管：拓樸變更與備份\n"
        "AI agent：自動稽核診斷（唯讀 ot/ 語料）\n"
        "簽核鏈：2/2 完成"
    ),
    "A.5.3": (
        "[HR + IAM 對帳]\n"
        "特權帳號與一般維運帳號分離；變更視窗需雙人覆核。\n"
        "本次抽樣 12 帳號，0 項 SoD 衝突。"
    ),
    "A.5.7": (
        "[GRC 威脅情報摘要]\n"
        "訂閱 ICS-CERT／CISA；本季 OT 相關公告 3 則已納入修補排程。"
    ),
}


def controls_needing_manual_import() -> list[dict]:
    """93 項 Annex A 中，非 OT syslog 自動監控者（需 AI 匯入／人工證據）。"""
    from code.services.evidence_chain_service import all_annex_controls_for_monitor

    return [c for c in all_annex_controls_for_monitor() if not c.get("monitored")]


def simulate_demo_batch(*, skip_existing: bool = True) -> dict:
    """
    示範批次：為所有需人工／GRC 證據的控制項各產生一筆模擬匯入。
    skip_existing：該 annex_a_id 已有任何匯入紀錄則略過（避免重複按鈕堆疊）。
    """
    targets = controls_needing_manual_import()
    have: set[str] = set()
    if skip_existing:
        for rec in list_imports(limit=10000):
            aid = (rec.get("annex_a_id") or "").strip()
            if aid:
                have.add(aid)

    created: list[dict] = []
    skipped: list[str] = []
    for c in targets:
        aid = (c.get("annex_a_id") or "").strip()
        if not aid:
            continue
        if skip_existing and aid in have:
            skipped.append(aid)
            continue
        title = (c.get("title") or aid).strip()
        created.append(
            append_import(
                annex_a_id=aid,
                title=title,
                content=_simulated_body_for_control(c),
                source="simulate_manual_batch",
                simulated=True,
                metadata={
                    "batch": "full_manual_coverage",
                    "pipeline": "ai_import",
                    "domain": c.get("domain"),
                },
            )
        )
        have.add(aid)

    return {
        "created": created,
        "skipped_annex_ids": skipped,
        "target_count": len(targets),
    }
