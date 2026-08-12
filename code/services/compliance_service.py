"""
ISO 27001:2022 Annex A 控制項矩陣與覆蓋率統計。

Phase 1 MVP：A.5 Organizational + A.8 Technological（共 71 項定義，50 項 in_scope）。
自動化監控：6 項 OT syslog 對映（見 monitored=true）。
"""
from __future__ import annotations

import json
from pathlib import Path

from code.paths import compliance_dir

COMPLIANCE_DIR = compliance_dir()
CONTROLS_PATH = COMPLIANCE_DIR / "annex_a_controls.json"

# A.5 Organizational (2022) — 37 項
_A5_TITLES = {
    "A.5.1": "資訊安全政策",
    "A.5.2": "資訊安全角色與責任",
    "A.5.3": "職責分離",
    "A.5.4": "管理階層責任",
    "A.5.5": "與主管機關聯絡",
    "A.5.6": "與特殊利益團體聯絡",
    "A.5.7": "威脅情報",
    "A.5.8": "專案管理中的資訊安全",
    "A.5.9": "資訊及其他相關資產清單",
    "A.5.10": "資訊及其他相關資產之可接受使用",
    "A.5.11": "資產歸還",
    "A.5.12": "資訊分類",
    "A.5.13": "資訊標示",
    "A.5.14": "資訊傳輸",
    "A.5.15": "存取控制",
    "A.5.16": "身分管理",
    "A.5.17": "驗證資訊",
    "A.5.18": "存取權限",
    "A.5.19": "供應商關係之資訊安全",
    "A.5.20": "供應商協議中資安要求",
    "A.5.21": "供應鏈中 ICT 之資訊安全管理",
    "A.5.22": "供應商服務監控、審查與變更",
    "A.5.23": "雲端服務使用之資訊安全",
    "A.5.24": "資訊安全事件管理規劃與準備",
    "A.5.25": "資訊安全事件評估與決策",
    "A.5.26": "資訊安全事件回應",
    "A.5.27": "從資訊安全事件學習",
    "A.5.28": "證據收集",
    "A.5.29": "中斷期間資訊安全",
    "A.5.30": "ICT 備援以確保持續營運",
    "A.5.31": "法律、法規及合約要求",
    "A.5.32": "智慧財產權",
    "A.5.33": "紀錄之保護",
    "A.5.34": "隱私及 PII 保護",
    "A.5.35": "資訊安全之獨立審查",
    "A.5.36": "資訊安全政策、規則及標準之遵循",
    "A.5.37": "已文件化操作程序",
}

# A.6 People (2022) — 8 項（矩陣 Phase 1 未納入定義）
_A6_TITLES = {
    "A.6.1": "篩選",
    "A.6.2": "雇用條款及條件",
    "A.6.3": "資訊安全意識、教育及訓練",
    "A.6.4": "懲戒程序",
    "A.6.5": "離職或職務異動後之責任",
    "A.6.6": "保密或不披露協議",
    "A.6.7": "遠距工作",
    "A.6.8": "資訊安全事件通報",
}

# A.7 Physical (2022) — 14 項（矩陣 Phase 1 未納入定義）
_A7_TITLES = {
    "A.7.1": "實體安全周界",
    "A.7.2": "實體進出",
    "A.7.3": "辦公室、房間及設施之安全",
    "A.7.4": "實體安全監控",
    "A.7.5": "對實體及環境威脅之防護",
    "A.7.6": "在安全區域內工作",
    "A.7.7": "清桌及清螢幕",
    "A.7.8": "設備安置及保護",
    "A.7.9": "場外資產之安全",
    "A.7.10": "儲存媒體",
    "A.7.11": "支援性設施",
    "A.7.12": "纜線安全",
    "A.7.13": "設備維護",
    "A.7.14": "設備之安全處置或再利用",
}

# A.8 Technological (2022) — 34 項
_A8_TITLES = {
    "A.8.1": "使用者端點裝置",
    "A.8.2": "特權存取權限",
    "A.8.3": "資訊存取限制",
    "A.8.4": "原始碼存取",
    "A.8.5": "安全驗證",
    "A.8.6": "容量管理",
    "A.8.7": "惡意軟體防護",
    "A.8.8": "技術弱點管理",
    "A.8.9": "組態管理",
    "A.8.10": "資訊刪除",
    "A.8.11": "資料遮罩",
    "A.8.12": "資料外洩防護",
    "A.8.13": "資訊備份",
    "A.8.14": "資訊處理設施之備援",
    "A.8.15": "記錄",
    "A.8.16": "監控活動",
    "A.8.17": "時鐘同步",
    "A.8.18": "特權公用程式的使用",
    "A.8.19": "作業系統上之軟體安裝",
    "A.8.20": "網路安全",
    "A.8.21": "網路服務之安全",
    "A.8.22": "網路分割",
    "A.8.23": "網路過濾",
    "A.8.24": "密碼學之使用",
    "A.8.25": "安全開發生命週期",
    "A.8.26": "應用程式安全需求",
    "A.8.27": "安全系統架構及工程原則",
    "A.8.28": "安全編碼",
    "A.8.29": "開發及測試環境之安全",
    "A.8.30": "委外開發",
    "A.8.31": "開發、測試及正式環境之分離",
    "A.8.32": "變更管理",
    "A.8.33": "測試資訊",
    "A.8.34": "稽核測試期間資訊系統之保護",
}

# 現行 OT syslog 自動對映（6 項）
_OT_MONITORED = {
    "A.5.19": {
        "control_key": "supplier_security",
        "evidence_types": ["syslog_cdp", "syslog_lldp", "remote_logging"],
        "wazuh_signal": "external_neighbor, remote_host",
    },
    "A.8.7": {
        "control_key": "malware_defense",
        "evidence_types": ["syslog_ips", "syslog_malware", "usb_alert"],
        "wazuh_signal": "malware, ips, host_attack",
    },
    "A.8.8": {
        "control_key": "patch_management",
        "evidence_types": ["syslog_ios_version", "upgrade_event"],
        "wazuh_signal": "software, image, smu",
    },
    "A.8.19": {
        "control_key": "recipe_audit",
        "evidence_types": ["syslog_config", "interface_change", "port_security"],
        "wazuh_signal": "config_i, parser, updown",
    },
    "A.8.24": {
        "control_key": "sec_gem_log",
        "evidence_types": ["syslog_snmp", "crypto_pki", "tls_session"],
        "wazuh_signal": "snmp, crypto, certificate",
    },
    "A.5.15": {
        "control_key": "access_control",
        "evidence_types": ["syslog_login", "radius", "tacacs", "ssh"],
        "wazuh_signal": "sec_login, aaa, auth",
        "legacy_annex_ref": "A.7.4",
        "note": "2022 存取控制主控項為 A.5.15；syslog 對映沿用 access_control key",
    },
}

# Phase 1 MVP in_scope：A.5 全項 + A.8 前 13 項（威脅情報到記錄）≈ 50
_PHASE1_A8_IN_SCOPE = {f"A.8.{i}" for i in range(1, 14)}


def _build_controls() -> list[dict]:
    rows: list[dict] = []
    for annex_id, title in _A5_TITLES.items():
        meta = _OT_MONITORED.get(annex_id, {})
        rows.append({
            "annex_a_id": annex_id,
            "title": title,
            "domain": "organizational",
            "phase1_mvp": True,
            "monitored": annex_id in _OT_MONITORED,
            "control_key": meta.get("control_key"),
            "evidence_types": meta.get("evidence_types", ["policy_doc", "procedure", "manual_review"]),
            "wazuh_signal": meta.get("wazuh_signal"),
            "automation": "syslog_regex" if annex_id in _OT_MONITORED else "manual",
        })
    for annex_id, title in _A8_TITLES.items():
        meta = _OT_MONITORED.get(annex_id, {})
        in_scope = annex_id in _PHASE1_A8_IN_SCOPE or annex_id in _OT_MONITORED
        rows.append({
            "annex_a_id": annex_id,
            "title": title,
            "domain": "technological",
            "phase1_mvp": in_scope,
            "monitored": annex_id in _OT_MONITORED,
            "control_key": meta.get("control_key"),
            "evidence_types": meta.get("evidence_types", ["config_review", "manual_test"]),
            "wazuh_signal": meta.get("wazuh_signal"),
            "automation": "syslog_regex" if annex_id in _OT_MONITORED else "manual",
        })
    return rows


def ensure_controls_file() -> Path:
    if not CONTROLS_PATH.is_file():
        payload = {
            "version": "2022",
            "standard": "ISO/IEC 27001:2022 Annex A",
            "principle": "No Evidence, No Compliance Claim",
            "total_annex_a": 93,
            "phase1_scope": "A.5 (37) + A.8 subset (13) = 50 MVP definitions",
            "controls": _build_controls(),
        }
        CONTROLS_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONTROLS_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return CONTROLS_PATH


def load_controls() -> list[dict]:
    ensure_controls_file()
    data = json.loads(CONTROLS_PATH.read_text(encoding="utf-8"))
    return data.get("controls") or []


def control_by_key(control_key: str) -> dict | None:
    for c in load_controls():
        if c.get("control_key") == control_key:
            return c
    return None


_DOMAIN_LABELS = {
    "organizational": "A.5 組織面",
    "people": "A.6 人員面",
    "physical": "A.7 實體面",
    "technological": "A.8 技術面",
}

# 未納入矩陣定義之 Annex A（A.6 + A.7）
_OUT_OF_MATRIX_META = {
    **{
        aid: {
            "domain": "people",
            "summary": "Phase 1 矩陣僅定義 A.5／A.8；人員面控制項尚未建檔。",
            "detail": (
                "ISO/IEC 27001:2022 Annex A 人員面控制，涵蓋到職篩選、"
                "資安訓練、離職責任與保密協議等。Semi-Shield 目前以 OT syslog "
                "自動化為主，人員面須以政策文件、HR 紀錄與人工稽核佐證。"
            ),
            "phase_plan": "Phase 2：納入 GRC 矩陣定義與 evidence 範本",
            "evidence_hint": "policy_doc · training_record · hr_offboarding · manual_review",
        }
        for aid in _A6_TITLES
    },
    **{
        aid: {
            "domain": "physical",
            "summary": "Phase 1 矩陣僅定義 A.5／A.8；實體面控制項尚未建檔。",
            "detail": (
                "實體進出、機房周界、清桌清螢幕、媒體處置等。"
                "OT 場域（無塵室／機櫃）須對照實體巡檢與門禁紀錄；"
                "本平台無法從 syslog 自動推論實體合規。"
            ),
            "phase_plan": "Phase 2：與實體門禁／CCTV 日誌整合（若可取得）",
            "evidence_hint": "access_badge_log · cctv_audit · facility_checklist · manual_review",
        }
        for aid in _A7_TITLES
    },
}


def _flashcard_from_control(c: dict, gap_type: str, **extra) -> dict:
    domain = c.get("domain") or "organizational"
    return {
        "annex_a_id": c.get("annex_a_id"),
        "title": c.get("title"),
        "domain": domain,
        "domain_label": _DOMAIN_LABELS.get(domain, domain),
        "gap_type": gap_type,
        "phase1_mvp": bool(c.get("phase1_mvp")),
        "monitored": bool(c.get("monitored")),
        "automation": c.get("automation") or "manual",
        "control_key": c.get("control_key"),
        "evidence_types": c.get("evidence_types") or [],
        **extra,
    }


def coverage_gaps() -> dict:
    """矩陣覆蓋率「未包含」控制項，供前端識字卡彈窗。"""
    controls = load_controls()
    defined_ids = {c.get("annex_a_id") for c in controls if c.get("annex_a_id")}

    not_in_matrix: list[dict] = []
    for aid, title in {**_A6_TITLES, **_A7_TITLES}.items():
        meta = _OUT_OF_MATRIX_META.get(aid, {})
        not_in_matrix.append({
            "annex_a_id": aid,
            "title": title,
            "domain": meta.get("domain", "people"),
            "domain_label": _DOMAIN_LABELS.get(meta.get("domain", "people"), ""),
            "gap_type": "not_in_matrix",
            "phase1_mvp": False,
            "monitored": False,
            "automation": "manual",
            "control_key": None,
            "evidence_types": (meta.get("evidence_hint") or "").split(" · "),
            "summary": meta.get("summary", "尚未納入 Semi-Shield 合規矩陣。"),
            "detail": meta.get("detail", ""),
            "phase_plan": meta.get("phase_plan", "Phase 2"),
            "evidence_hint": meta.get("evidence_hint", "manual_review"),
        })
    not_in_matrix.sort(key=lambda x: x["annex_a_id"])

    phase2_in_matrix: list[dict] = []
    not_automated: list[dict] = []
    for c in controls:
        aid = c.get("annex_a_id")
        if not c.get("phase1_mvp"):
            phase2_in_matrix.append(_flashcard_from_control(
                c,
                "phase2_in_matrix",
                summary="已建矩陣定義，但列為 Phase 2（非 MVP in_scope）。",
                detail=(
                    f"控制項 {aid} 已在 A.5／A.8 矩陣中定義，"
                    "尚未列入 Phase 1 MVP 自動化範圍。"
                    "可透過人工 evidence 或 Phase 2 擴充監控。"
                ),
                phase_plan="Phase 2 MVP 擴充或維持人工稽核",
                evidence_hint=" · ".join(c.get("evidence_types") or ["manual_review"]),
            ))
        if not c.get("monitored"):
            not_automated.append(_flashcard_from_control(
                c,
                "not_automated",
                summary="矩陣已涵蓋，但尚無 OT syslog 自動對映。",
                detail=(
                    "Semi-Shield Phase 1 僅 6 項 syslog 自動監控；"
                    "其餘控制項需 policy／config review 或人工註記 evidence。"
                ),
                phase_plan="Phase 2：Wazuh 規則擴充至 15+ 項" if c.get("domain") == "technological" else "維持文件化 evidence",
                evidence_hint=" · ".join(c.get("evidence_types") or ["manual_review"]),
            ))

    phase2_in_matrix.sort(key=lambda x: x["annex_a_id"])
    not_automated.sort(key=lambda x: x["annex_a_id"])

    total_annex = 93
    return {
        "annex_a_total": total_annex,
        "matrix_defined": len(defined_ids),
        "not_in_matrix_count": len(not_in_matrix),
        "phase2_in_matrix_count": len(phase2_in_matrix),
        "not_automated_count": len(not_automated),
        "not_in_matrix": not_in_matrix,
        "phase2_in_matrix": phase2_in_matrix,
        "not_automated": not_automated,
    }


def coverage_stats() -> dict:
    controls = load_controls()
    total_annex = 93
    defined = len(controls)
    phase1 = sum(1 for c in controls if c.get("phase1_mvp"))
    monitored = sum(1 for c in controls if c.get("monitored"))
    automated = sum(1 for c in controls if c.get("automation") == "syslog_regex")
    gaps = coverage_gaps()
    return {
        "principle": "No Evidence, No Compliance Claim",
        "annex_a_total": total_annex,
        "matrix_defined": defined,
        "phase1_mvp_count": phase1,
        "automated_monitoring_count": automated,
        "monitored_control_keys": [
            c["control_key"] for c in controls if c.get("monitored") and c.get("control_key")
        ],
        "coverage_rate_matrix": round(defined / total_annex, 4),
        "phase1_coverage_rate": round(phase1 / total_annex, 4),
        "automated_coverage_rate": round(automated / total_annex, 4),
        "target_annex_coverage": 0.8,
        "target_phase1_note": "Phase 1 目標 ≥80% Annex A 有定義；自動化監控 6→15 項為 Phase 2",
        "not_in_matrix_count": gaps["not_in_matrix_count"],
        "phase2_in_matrix_count": gaps["phase2_in_matrix_count"],
        "not_automated_count": gaps["not_automated_count"],
    }


def load_json(name: str) -> dict:
    path = COMPLIANCE_DIR / name
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_status(
    current: float | int | None,
    target: float | int,
    *,
    higher_is_better: bool = True,
) -> str:
    if current is None:
        return "na"
    try:
        ok = current >= target if higher_is_better else current <= target
    except TypeError:
        return "na"
    return "ok" if ok else "warn"


def _analysis_item(
    metric_id: str,
    *,
    category: str,
    title: str,
    view: str = "analysis",
    status: str = "na",
    current_display: str = "—",
    target_display: str = "—",
    summary: str = "",
    formula: str = "",
    detail: str = "",
    breakdown: list | None = None,
    recommendations: list | None = None,
    flashcard_tab: str | None = None,
    extra: dict | None = None,
) -> dict:
    item = {
        "id": metric_id,
        "category": category,
        "title": title,
        "view": view,
        "status": status,
        "current_display": current_display,
        "target_display": target_display,
        "summary": summary,
        "formula": formula,
        "detail": detail,
        "breakdown": breakdown or [],
        "recommendations": recommendations or [],
        "flashcard_tab": flashcard_tab,
    }
    if extra:
        item.update(extra)
    return item


def build_metrics_analysis(
    traceability: dict | None = None,
    human_review: dict | None = None,
    evidence_items: list | None = None,
) -> dict:
    """各項 KPI／覆蓋率／Evidence／監控指標的詳細分析（供前端彈窗）。"""
    trace = traceability or {}
    review = human_review or {}
    evidence = evidence_items or []

    cov = coverage_stats()
    gaps = coverage_gaps()
    controls = load_controls()
    kpi_cfg = load_json("kpi_targets.json")
    kpi_targets = kpi_cfg.get("targets") or {}
    measurements = kpi_cfg.get("measurement") or {}

    matrix_rate = cov.get("coverage_rate_matrix") or 0
    phase1_rate = cov.get("phase1_coverage_rate") or 0
    auto_rate = cov.get("automated_coverage_rate") or 0
    auto_count = cov.get("automated_monitoring_count") or 0
    trace_rate = trace.get("traceability_rate")
    adoption = review.get("adoption_rate")
    mapping_accuracy = review.get("control_mapping_accuracy")
    by_type = review.get("by_type") or {}
    cs_stats = by_type.get("control_status") or {}
    gr_stats = by_type.get("guardrail_block") or {}

    # Evidence 依 control_key 分組
    ev_by_key: dict[str, list] = {}
    for row in evidence:
        key = row.get("control_key") or row.get("annex_a_id") or "unknown"
        ev_by_key.setdefault(str(key), []).append(row)

    items: list[dict] = []

    # --- KPI ---
    annex_target = (kpi_targets.get("annex_a_coverage_rate") or {}).get("target", 0.8)
    items.append(_analysis_item(
        "kpi_annex_a_coverage_rate",
        category="kpi",
        title=(kpi_targets.get("annex_a_coverage_rate") or {}).get("label", "Annex A 矩陣覆蓋率"),
        status=_metric_status(matrix_rate, annex_target),
        current_display=f"{round(matrix_rate * 100)}%",
        target_display=f"≥{round(annex_target * 100)}%",
        summary=(
            f"已定義 {cov.get('matrix_defined', 0)} / {cov.get('annex_a_total', 93)} 項 Annex A；"
            f"缺口 {gaps.get('not_in_matrix_count', 0)} 項（A.6／A.7）尚未納入矩陣。"
        ),
        formula="matrix_defined ÷ 93（Annex A 2022 全項）",
        detail=(kpi_targets.get("annex_a_coverage_rate") or {}).get("note", ""),
        breakdown=[
            {"label": "矩陣已定義", "value": str(cov.get("matrix_defined", 0)), "note": "A.5 + A.8"},
            {"label": "Annex A 總計", "value": "93", "note": "2022 版四域"},
            {"label": "未納入矩陣", "value": str(gaps.get("not_in_matrix_count", 0)), "note": "A.6 人員 + A.7 實體"},
            {"label": "Phase 1 MVP", "value": str(cov.get("phase1_mvp_count", 0)), "note": "in_scope 定義"},
        ],
        recommendations=[
            "Phase 2 將 A.6／A.7 納入矩陣定義與 evidence 範本",
            "點擊下方「未納入矩陣」識字卡檢視 22 項缺口",
        ],
        flashcard_tab="not_in_matrix",
    ))

    auto_target = (kpi_targets.get("automated_monitoring_coverage") or {}).get("target", 6)
    try:
        from code.services.ai_reviewer_service import ai_reviewer_enabled
        ai_rev = ai_reviewer_enabled()
    except Exception:
        ai_rev = False

    items.append(_analysis_item(
        "kpi_automated_monitoring_coverage",
        category="kpi",
        title=(kpi_targets.get("automated_monitoring_coverage") or {}).get("label", "自動化監控覆蓋"),
        status=_metric_status(auto_count, auto_target),
        current_display=f"{auto_count} 項",
        target_display=f"≥{auto_target} 項",
        summary=f"現行 {auto_count} 項 OT syslog 正規表示式對映；Phase 2 目標 15+ 項。",
        formula="monitored=true 且 automation=syslog_regex 的控制項數",
        detail=(kpi_targets.get("automated_monitoring_coverage") or {}).get("note", ""),
        breakdown=[
            {"label": "自動化對映", "value": str(auto_count), "note": "佔 Annex A 93 項之 " + f"{round(auto_rate * 100, 1)}%"},
            {"label": "未自動化（矩陣內）", "value": str(gaps.get("not_automated_count", 0)), "note": "需人工 evidence"},
            {"label": "control_keys", "value": ", ".join(cov.get("monitored_control_keys") or []), "note": "OT 監控鍵"},
        ],
        recommendations=["Phase 2 擴充 Wazuh 規則至 15+ 控制項", "優先擴 A.8.16 監控活動、A.8.20 網路安全"],
        flashcard_tab="not_automated",
    ))

    map_target = review.get("target_control_mapping_accuracy") or (
        (kpi_targets.get("control_mapping_accuracy") or {}).get("target", 0.85)
    )
    cs_decided = (cs_stats.get("approved") or 0) + (cs_stats.get("rejected") or 0)
    map_display = f"{round(mapping_accuracy * 100)}%" if mapping_accuracy is not None else "待累積樣本"
    map_summary = (
        "僅統計 control_status 項目：AI Reviewer 依 control_key／Annex A 對映與 evidence 追溯核准的比例。"
        if ai_rev
        else "control_status 項目經分析師覆核且 Annex A 標註正確之比例。"
    )
    items.append(_analysis_item(
        "kpi_control_mapping_accuracy",
        category="kpi",
        title=(kpi_targets.get("control_mapping_accuracy") or {}).get("label", "控制項對映準確率"),
        status=_metric_status(mapping_accuracy, map_target) if mapping_accuracy is not None else "na",
        current_display=map_display,
        target_display=f"≥{round(map_target * 100)}%",
        summary=map_summary,
        formula="control_status 核准 / (control_status 核准 + 駁回)",
        detail=(kpi_targets.get("control_mapping_accuracy") or {}).get("note", ""),
        breakdown=[
            {"label": "control_status 已審", "value": str(cs_decided), "note": "不含 guardrail_block"},
            {"label": "核准", "value": str(cs_stats.get("approved") or 0), "note": "Annex A 對映通過"},
            {"label": "駁回", "value": str(cs_stats.get("rejected") or 0), "note": "缺 evidence 或對映"},
            {"label": "護欄攔截", "value": str(gr_stats.get("rejected") or 0), "note": "不計入本 KPI"},
        ],
        recommendations=(
            [
                "若數值偏低，多為舊 queue 在 evidence 補齊前被駁回；點「AI 自動審核」會 reconcile",
                "AI 仍會驗證 annex_a_id 與 control_key",
            ]
            if ai_rev
            else ["於 Review Queue 核准時確認 annex_a_id 與 control_key 正確", "累積 20+ 筆已審核後指標才有統計意義"]
        ),
    ))

    adopt_target = review.get("target_adoption_rate") or (kpi_targets.get("human_review_adoption_rate") or {}).get("target", 0.75)
    adopt_display = f"{round(adoption * 100)}%" if adoption is not None else "待累積樣本"
    hr_summary = (
        "由 AI Reviewer 依 evidence／hash chain 自動核准或駁回（無需分析師）。"
        if ai_rev
        else "分析師核准 LLM 診斷／護欄邊界項目之比例。"
    )
    hr_recs = (
        ["點擊「AI 自動審核」處理 pending 項目", "護欄攔截項目維持 rejected"]
        if ai_rev
        else ["定期清空 pending 項目", "駁回時註記原因供模型／規則改善"]
    )
    if ai_rev and (review.get("pending") or 0) > 0:
        hr_recs.insert(0, f"目前有 {review.get('pending')} 筆待審，可一鍵 AI 審核")
    items.append(_analysis_item(
        "kpi_human_review_adoption_rate",
        category="kpi",
        title=(kpi_targets.get("human_review_adoption_rate") or {}).get("label", "Human review 採納率"),
        status=_metric_status(adoption, adopt_target) if adoption is not None else "na",
        current_display=adopt_display,
        target_display=f"≥{round(adopt_target * 100)}%",
        summary=hr_summary,
        formula=measurements.get("human_review_adoption_rate", "approved / (approved + rejected)"),
        detail=(kpi_targets.get("human_review_adoption_rate") or {}).get("note", ""),
        breakdown=[
            {"label": "合規審核已決", "value": str((review.get("operational") or {}).get("decided") or 0), "note": "control_status + llm_diagnosis"},
            {"label": "核准", "value": str((review.get("operational") or {}).get("approved") or 0), "note": "含 ai-reviewer" if ai_rev else ""},
            {"label": "駁回", "value": str((review.get("operational") or {}).get("rejected") or 0), "note": "不含護欄攔截"},
            {"label": "護欄攔截（另計）", "value": str(gr_stats.get("rejected") or 0), "note": "惡意請求維持 rejected"},
        ],
        recommendations=hr_recs,
    ))

    ev_target = (kpi_targets.get("evidence_traceability_rate") or {}).get("target", 1.0)
    ev_display = f"{round(trace_rate * 100)}%" if trace_rate is not None and trace.get("total_evidence_records") else "—"
    items.append(_analysis_item(
        "kpi_evidence_traceability_rate",
        category="kpi",
        title=(kpi_targets.get("evidence_traceability_rate") or {}).get("label", "Evidence 可追溯率"),
        status=_metric_status(trace_rate, ev_target) if trace.get("total_evidence_records") else "na",
        current_display=ev_display,
        target_display="100%",
        summary="每項合規宣稱須具 evidence_id 與 hash chain（No Evidence, No Compliance Claim）。",
        formula=measurements.get("evidence_traceability_rate", "含 evidence_id / 全部 evidence"),
        detail=(kpi_targets.get("evidence_traceability_rate") or {}).get("note", ""),
        breakdown=[
            {"label": "Evidence 紀錄", "value": str(trace.get("total_evidence_records") or 0), "note": ""},
            {"label": "含 evidence_id", "value": str(trace.get("with_evidence_id") or 0), "note": ""},
            {"label": "含 payload_hash", "value": str(trace.get("with_payload_hash") or 0), "note": "hash chain"},
            {"label": "待審 evidence", "value": str(trace.get("pending_review") or 0), "note": ""},
        ],
        recommendations=["執行合規管線寫入 evidence", "核准 pending evidence 後方可納入稽核報告"],
        flashcard_tab=None,
    ))

    rpt_cfg = kpi_targets.get("report_generation_time_days") or {}
    items.append(_analysis_item(
        "kpi_report_generation_time_days",
        category="kpi",
        title=rpt_cfg.get("label", "合規報告生成時間"),
        status="ok",
        current_display=f"≤{rpt_cfg.get('target', 2)} 天",
        target_display=f"≤{rpt_cfg.get('target', 2)} 天（基線 {rpt_cfg.get('baseline_days', 28)} 天）",
        summary="地端 Agent 管線：掃描 OT → LLM 診斷 → Review → PDF／TXT 匯出。",
        formula="管線完成時間 + 分析師覆核時間",
        detail=rpt_cfg.get("note", ""),
        breakdown=[
            {"label": "目標", "value": f"{rpt_cfg.get('target', 2)} 天", "note": "Semi-Shield"},
            {"label": "傳統基線", "value": f"{rpt_cfg.get('baseline_days', 28)} 天", "note": "顧問人工"},
            {"label": "匯出路徑", "value": "監控戰情室", "note": "PDF / TXT"},
        ],
        recommendations=["先執行管線再至監控戰情室匯出報告", "LLM 診斷須經 Human Review 後方可作稽核證據"],
    ))

    cost_cfg = kpi_targets.get("compliance_cost_reduction") or {}
    items.append(_analysis_item(
        "kpi_compliance_cost_reduction",
        category="kpi",
        title=cost_cfg.get("label", "合規評估成本降低"),
        status="ok",
        current_display=f"≥{round((cost_cfg.get('target') or 0.7) * 100)}%",
        target_display=f"≥{round((cost_cfg.get('target') or 0.7) * 100)}%",
        summary="相較純人工顧問整理 syslog 證據與矩陣對照之預估節省。",
        formula="（顧問人天 − 地端自動化人天）/ 顧問人天",
        detail=cost_cfg.get("note", ""),
        breakdown=[
            {"label": "自動化", "value": f"{auto_count} 控制項", "note": "syslog 對映"},
            {"label": "Agent 管線", "value": "4 階段", "note": "Collector→Reporter"},
            {"label": "地端 LLM", "value": "Ollama", "note": "無雲端 API 費"},
        ],
        recommendations=["擴大自動監控覆蓋可進一步降低人工對照成本"],
    ))

    # --- Coverage cards ---
    items.append(_analysis_item(
        "cov_matrix_defined",
        category="coverage",
        title="矩陣定義",
        status="ok",
        current_display=f"{cov.get('matrix_defined', 0)} / 93",
        target_display="≥57（80%）",
        summary="Semi-Shield 合規矩陣已建檔之 Annex A 控制項數（目前 A.5 全項 + A.8 全項）。",
        formula="len(controls in annex_a_controls.json)",
        detail="不含 A.6 人員面與 A.7 實體面共 22 項。",
        breakdown=[
            {"label": "A.5 組織面", "value": "37", "note": "全數定義"},
            {"label": "A.8 技術面", "value": "34", "note": "全數定義"},
            {"label": "A.6 + A.7", "value": "0", "note": f"缺口 {gaps.get('not_in_matrix_count', 22)} 項"},
        ],
        flashcard_tab="not_in_matrix",
    ))

    items.append(_analysis_item(
        "cov_phase1_mvp",
        category="coverage",
        title="Phase 1 MVP",
        status=_metric_status(phase1_rate, 0.538),  # ~50/93
        current_display=f"{cov.get('phase1_mvp_count', 0)} 項",
        target_display="≈50 項 in_scope",
        summary="Phase 1 優先實施範圍：A.5 全項 + A.8.1–A.8.13，另含 OT 監控對映項。",
        formula="phase1_mvp=true 的控制項數",
        detail=cov.get("target_phase1_note", ""),
        breakdown=[
            {"label": "MVP 佔 Annex A", "value": f"{round(phase1_rate * 100, 1)}%", "note": ""},
            {"label": "Phase 2 定義", "value": str(gaps.get("phase2_in_matrix_count", 0)), "note": "矩陣內非 MVP"},
        ],
        flashcard_tab="phase2_in_matrix",
    ))

    items.append(_analysis_item(
        "cov_automated_monitoring",
        category="coverage",
        title="OT 自動監控",
        status=_metric_status(auto_count, 6),
        current_display=f"{auto_count} 項",
        target_display="6 項（Phase 1）",
        summary="Cisco syslog 正規表示式自動對映至 Annex A 與 control_key。",
        formula="monitored=true 之控制項",
        detail="對映：A.5.15、A.5.19、A.8.7、A.8.8、A.8.19、A.8.24",
        breakdown=[
            {"label": "佔 Annex A", "value": f"{round(auto_rate * 100, 1)}%", "note": "93 項中"},
            {"label": "未自動化", "value": str(gaps.get("not_automated_count", 0)), "note": "點擊識字卡"},
        ],
        flashcard_tab="not_automated",
    ))

    items.append(_analysis_item(
        "cov_matrix_rate",
        category="coverage",
        title="矩陣覆蓋率",
        view="analysis",
        status=_metric_status(matrix_rate, annex_target),
        current_display=f"{round(matrix_rate * 100)}%",
        target_display=f"≥{round(annex_target * 100)}%",
        summary=f"距離 80% 目標差 {max(0, round(annex_target * 93) - cov.get('matrix_defined', 0))} 項定義（需納入 A.6／A.7 或擴展矩陣）。",
        formula="matrix_defined / 93",
        detail="點擊下方分頁檢視未覆蓋控制項識字卡。",
        flashcard_tab="not_in_matrix",
    ))

    # --- Evidence cards ---
    total_ev = trace.get("total_evidence_records") or 0
    items.append(_analysis_item(
        "ev_total_records",
        category="evidence",
        title="Evidence 紀錄",
        status="ok" if total_ev > 0 else "na",
        current_display=str(total_ev),
        target_display="≥6（每監控控制項 1 筆）",
        summary="evidence_registry.jsonl 已寫入之 Collector 紀錄總數。",
        formula="count(evidence_registry.jsonl)",
        detail="由合規管線或監控掃描產生，每筆含 payload_hash 鏈。",
        breakdown=[
            {"label": "控制項分組", "value": str(len(ev_by_key)), "note": "distinct control_key"},
            *[
                {"label": k, "value": str(len(v)), "note": (v[-1].get("evidence_id") or "")[:24]}
                for k, v in sorted(ev_by_key.items())[:8]
            ],
        ],
        recommendations=["執行「地端合規管線」刷新 evidence", "待審項目請至 Review Queue 核准"],
    ))

    items.append(_analysis_item(
        "ev_with_evidence_id",
        category="evidence",
        title="含 evidence_id",
        status=_metric_status(
            trace.get("with_evidence_id"),
            total_ev if total_ev else 1,
        ) if total_ev else "na",
        current_display=str(trace.get("with_evidence_id") or 0),
        target_display=str(total_ev or "—"),
        summary="具唯一 EV- 編號、可引用於稽核報告之紀錄數。",
        formula="rows where evidence_id != null",
        detail="格式：EV-ot-fab-{control_key}-{timestamp}-{hash}",
        breakdown=[
            {"label": "總紀錄", "value": str(total_ev), "note": ""},
            {"label": "缺 ID", "value": str(max(0, total_ev - (trace.get("with_evidence_id") or 0))), "note": "不應出現"},
        ],
    ))

    items.append(_analysis_item(
        "ev_traceability_rate",
        category="evidence",
        title="追溯率",
        status=_metric_status(trace_rate, ev_target) if total_ev else "na",
        current_display=ev_display,
        target_display="100%",
        summary="max(evidence_id 覆蓋率, payload_hash 覆蓋率)。",
        formula="max(with_id/total, with_hash/total)",
        detail="hash chain 確保 evidence 未被篡改。",
        breakdown=[
            {"label": "with_evidence_id", "value": str(trace.get("with_evidence_id") or 0), "note": ""},
            {"label": "with_payload_hash", "value": str(trace.get("with_payload_hash") or 0), "note": ""},
        ],
    ))

    items.append(_analysis_item(
        "ev_pending_review",
        category="evidence",
        title="待審 Evidence",
        status="warn" if (trace.get("pending_review") or 0) > 0 else "ok",
        current_display=str(trace.get("pending_review") or 0),
        target_display="0",
        summary="review_status=pending 之 evidence，須分析師核准後可作稽核證據。",
        formula="count(review_status=pending)",
        detail="與 Human Review Queue 搭配；核准後更新 review_status。",
        breakdown=[
            {"label": "待審", "value": str(trace.get("pending_review") or 0), "note": ""},
            {"label": "Review Queue 待審", "value": str(review.get("pending") or 0), "note": "含非 evidence 項目"},
        ],
        recommendations=["至 Review Queue 逐筆核准或駁回"],
    ))

    # --- Monitored overview + per control ---
    monitored_rows = [c for c in controls if c.get("monitored")]
    items.append(_analysis_item(
        "mon_overview",
        category="monitored",
        title="OT 自動監控對映總覽",
        status="ok" if monitored_rows else "na",
        current_display=f"{len(monitored_rows)} 項",
        target_display="6 項",
        summary="syslog 正規表示式對映至 Wazuh／平台 control_key。",
        formula="controls where monitored=true",
        detail="下表各列可點擊查看單一控制項信號與 evidence 類型。",
        breakdown=[
            {
                "label": c.get("annex_a_id") or "",
                "value": c.get("control_key") or "—",
                "note": c.get("title") or "",
            }
            for c in monitored_rows
        ],
    ))

    for c in monitored_rows:
        key = c.get("control_key") or ""
        meta = _OT_MONITORED.get(c.get("annex_a_id") or "", {})
        ev_rows = ev_by_key.get(key, [])
        latest = ev_rows[-1] if ev_rows else {}
        items.append(_analysis_item(
            f"mon_{key}",
            category="monitored",
            title=f"{c.get('annex_a_id')} {c.get('title')}",
            status="ok",
            current_display=key,
            target_display="syslog_regex",
            summary=f"Wazuh 信號：{meta.get('wazuh_signal') or c.get('wazuh_signal') or '—'}",
            formula="OT 日誌 regex → control_key metric",
            detail=meta.get("note") or "自動彙整 Cisco syslog 事件量與 pass/fail/review 狀態。",
            breakdown=[
                {"label": "Annex A", "value": c.get("annex_a_id") or "", "note": ""},
                {"label": "Evidence 類型", "value": ", ".join(c.get("evidence_types") or []), "note": ""},
                {"label": "Evidence 筆數", "value": str(len(ev_rows)), "note": ""},
                {"label": "最近事件量", "value": str(latest.get("event_count") or "—"), "note": latest.get("evidence_id") or ""},
            ],
            recommendations=["至監控戰情室查看即時計數與日誌樣本"],
        ))

    by_id = {x["id"]: x for x in items}
    return {
        "principle": "No Evidence, No Compliance Claim",
        "generated_from": {
            "coverage": cov,
            "traceability": trace,
            "human_review": review,
            "evidence_sample_count": len(evidence),
        },
        "items": items,
        "by_id": by_id,
        "coverage_gaps": gaps,
    }
