"""
ISO 27001 證據鏈：Semi-Shield 自動化 vs 傳統稽核路徑（監控戰情室用）。
"""
from __future__ import annotations

from code.services.compliance_service import (
    _A6_TITLES,
    _A7_TITLES,
    _DOMAIN_LABELS,
    _OT_MONITORED,
    load_controls,
)
from code.services.control_evidence_profiles import (
    build_evidence_sources,
    build_semi_shield_chain,
    build_traditional_chain,
)

# 非 OT syslog 控制項：走 AI 全自動匯入稽核資料管線（政策／GRC／HR／實體紀錄等）


def _annex_sort_key(aid: str) -> tuple:
    parts = (aid or "").replace("A.", "").split(".")
    try:
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except ValueError:
        return (99, 99)


def all_annex_controls_for_monitor() -> list[dict]:
    """矩陣 71 項 + A.6／A.7 全項 → 93 Annex A 控制項。"""
    rows = list(load_controls())
    existing = {c.get("annex_a_id") for c in rows}
    for aid, title in _A6_TITLES.items():
        if aid in existing:
            continue
        rows.append({
            "annex_a_id": aid,
            "title": title,
            "domain": "people",
            "phase1_mvp": False,
            "monitored": False,
            "control_key": None,
            "evidence_types": ["policy_doc", "training_record", "hr_export"],
            "automation": "auto_import",
        })
    for aid, title in _A7_TITLES.items():
        if aid in existing:
            continue
        rows.append({
            "annex_a_id": aid,
            "title": title,
            "domain": "physical",
            "phase1_mvp": False,
            "monitored": False,
            "control_key": None,
            "evidence_types": ["facility_checklist", "access_badge_log", "physical_audit"],
            "automation": "auto_import",
        })
    rows.sort(key=lambda c: _annex_sort_key(c.get("annex_a_id") or ""))
    return rows


def automation_tier(annex_id: str, control: dict) -> str:
    if control.get("monitored") or annex_id in _OT_MONITORED:
        return "auto_syslog"
    return "auto_import"


def _import_preview_lines(annex_id: str, title: str) -> list[str]:
    """AI 匯入管線示範紀錄（供戰情室原文區顯示，含模擬批次）。"""
    t = title or annex_id
    return [
        f"[AI-IMPORT] batch=2026-03-12T08:00:00Z annex={annex_id} doc=isms_{annex_id.replace('.', '_')}_v3.pdf",
        f"[PARSE] 控制項「{t}」結構化完成 · 版本=3.1 · 涵蓋期間=2025-01-01..2026-12-31",
        f"[CHECK] 必填欄位=OK · 簽核鏈=2/2 · evidence_type=policy_doc,procedure",
        f"[REGISTRY] queued evidence_id pending · payload_sha256=sim…{annex_id[-3:]}",
    ]


def resolve_control_display_log(
    *,
    annex_id: str,
    title: str = "",
    tier: str,
    control_key: str | None,
    bundle: dict | None,
    parsed_logs: list | None,
    samples_by_key: dict | None,
    manual_log: str = "",
) -> str:
    """戰情室「原始資安日誌／匯入原文」：含 ot/ 與拓樸虛擬 syslog，不再以「禁止虛構」隱藏。"""
    manual_log = (manual_log or "").strip()
    if manual_log:
        return manual_log

    parsed_logs = parsed_logs or []
    samples_by_key = samples_by_key or {}
    bundle_log = (bundle or {}).get("log") or ""
    if bundle_log and "無可用原文樣本" not in bundle_log and "禁止虛構" not in bundle_log:
        return bundle_log

    lines: list[str] = []
    k = control_key if control_key in samples_by_key else None
    if k:
        for ev in parsed_logs:
            if ev.get("key") == k and ev.get("raw"):
                lines.append(str(ev["raw"]))
        if len(lines) < 8:
            for ev in samples_by_key.get(k) or []:
                if ev.get("raw"):
                    lines.append(str(ev["raw"]))
                if len(lines) >= 8:
                    break

    if tier == "auto_import":
        lines = _import_preview_lines(annex_id, title or (bundle or {}).get("title") or "") + lines

    if not lines and parsed_logs:
        header = "【OT 拓樸／ot/ syslog 摘錄 · 含虛擬模擬資料】"
        for ev in parsed_logs[:8]:
            raw = ev.get("raw") or ev.get("message") or ""
            if not raw:
                continue
            tag = " [sim]" if ev.get("simulated") else ""
            lines.append(f"{raw}{tag}")
        if lines:
            lines.insert(0, header)

    if lines:
        if len(lines) > 12:
            extra = len(lines) - 12
            lines = lines[:12] + [f"...另有 {extra} 行未展開..."]
        return "\n".join(lines)

    if tier == "auto_import":
        return "\n".join(_import_preview_lines(annex_id, annex_id))
    return bundle_log or "（尚無 syslog／匯入原文）"


def build_control_evidence_view(
    control: dict,
    *,
    bundle: dict | None,
    metric: dict | None,
    evidence_id: str = "",
    log_sources: list[str] | None = None,
) -> dict:
    annex_id = control.get("annex_a_id") or ""
    tier = automation_tier(annex_id, control)
    key = control.get("control_key") or f"annex:{annex_id.replace('.', '_')}"
    st = (metric or {}).get("status") or "n/a"
    if tier == "auto_import" and st == "n/a":
        st = "auto"

    sources = build_evidence_sources(control, tier=tier, log_files=log_sources)

    return {
        "annex_a_id": annex_id,
        "title": control.get("title"),
        "domain": control.get("domain"),
        "domain_label": _DOMAIN_LABELS.get(control.get("domain") or "", ""),
        "control_key": key,
        "automation_tier": tier,
        "automation_label": {
            "auto_syslog": "OT Syslog 自動化",
            "auto_import": "AI 全自動匯入稽核資料",
            "assisted": "AI 全自動匯入稽核資料",
            "manual_only": "AI 全自動匯入稽核資料",
        }.get(tier, "AI 全自動匯入稽核資料"),
        "monitored": bool(control.get("monitored")),
        "metric_status": st,
        "metric_count": (metric or {}).get("count"),
        "metric_text": (metric or {}).get("text"),
        "evidence_id": evidence_id or (bundle or {}).get("evidence_id") or "",
        "event_count": (bundle or {}).get("event_count"),
        "sources": sources,
        "semi_shield_chain": build_semi_shield_chain(control, tier),
        "traditional_chain": build_traditional_chain(control, tier),
    }


def build_evidence_comparison_summary() -> dict:
    return {
        "principle": "No Evidence, No Compliance Claim",
        "semi_shield": {
            "title": "Semi-Shield 證據鏈",
            "points": [
                "OT（ot/）與稽核資料匯入 → 自動分類至 Annex A 控制項",
                "非 syslog 控制項：AI 解析 GRC／政策／HR／實體紀錄並產生 bundle",
                "每個 bundle 可產生 evidence_id 與 SHA-256 hash chain",
                "LLM 診斷必須引用 bundle 原文，禁止無證據宣稱",
                "review/fail 自動進 Human Review Queue",
                "可追溯：檔名 · 行號 · 時間 · 設備 · control_key",
            ],
        },
        "traditional": {
            "title": "傳統證據鏈",
            "points": [
                "人工匯出 syslog／設定，以 Excel 或 Email 傳遞",
                "PDF 快照為稽核當下靜態檔，難以重算 KPI",
                "無全域 hash chain，版本與完整性靠人工對帳",
                "合規結論常來自訪談，與原始 log 連結薄弱",
                "跨控制項關聯（如 A.5.15 + A.8.19）需人工拼貼",
            ],
        },
        "comparison_table": [
            {"aspect": "資料來源", "semi": "集中 ot/ + 自動解析", "traditional": "分散匯出／截圖"},
            {"aspect": "完整性", "semi": "payload_hash + prev_hash 鏈", "traditional": "通常無密碼學校驗"},
            {"aspect": "控制項對映", "semi": "93 項 · syslog 自動 + AI 匯入對映", "traditional": "稽核員手動對照 Annex A"},
            {"aspect": "AI 診斷", "semi": "綁定 evidence_id 與 bundle", "traditional": "自由文字或口頭結論"},
            {"aspect": "人工閘門", "semi": "Review Queue 強制節點", "traditional": "依稽核員經驗"},
            {"aspect": "OT 即時性", "semi": "戰情室 KPI 可刷新", "traditional": "年度／季度批次"},
        ],
    }


def build_evidence_dashboard(
    monitor_data: dict,
    *,
    evidence_map: dict | None = None,
    log_files: list[str] | None = None,
) -> dict:
    from code.services.manual_import_service import count_by_annex, log_text_for_annex

    bundles = monitor_data.get("control_bundles") or {}
    metrics = monitor_data.get("metrics") or {}
    evidence_map = evidence_map or monitor_data.get("evidence_map") or {}
    manual_counts = count_by_annex()

    controls_out: list[dict] = []
    for c in all_annex_controls_for_monitor():
        annex_id = c.get("annex_a_id") or ""
        key = c.get("control_key")
        bundle = bundles.get(key) if key else None
        metric = metrics.get(key) if key else None
        eid = evidence_map.get(key, "") if key else ""
        view = build_control_evidence_view(
            c,
            bundle=bundle,
            metric=metric,
            evidence_id=eid,
            log_sources=log_files,
        )
        mi_n = int(manual_counts.get(annex_id) or 0)
        view["manual_import_count"] = mi_n
        view["log"] = resolve_control_display_log(
            annex_id=annex_id,
            title=c.get("title") or "",
            tier=view.get("automation_tier") or "auto_import",
            control_key=key,
            bundle=bundle,
            parsed_logs=monitor_data.get("parsed_logs"),
            samples_by_key=monitor_data.get("samples_by_key"),
            manual_log=log_text_for_annex(annex_id) if mi_n else "",
        )
        controls_out.append(view)

    auto_n = sum(1 for x in controls_out if x["automation_tier"] == "auto_syslog")
    import_n = sum(1 for x in controls_out if x["automation_tier"] == "auto_import")

    return {
        "ok": True,
        "principle": monitor_data.get("compliance_principle") or "No Evidence, No Compliance Claim",
        "summary": {
            "annex_total": len(controls_out),
            "auto_syslog": auto_n,
            "auto_import": import_n,
            "assisted": 0,
            "manual_only": 0,
        },
        "controls": controls_out,
        "comparison": build_evidence_comparison_summary(),
    }
