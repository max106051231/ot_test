"""各 Annex A 控制項專屬證據來源與證據鏈（Semi-Shield vs 傳統）。"""
from __future__ import annotations

from code.services.compliance_service import _OT_MONITORED

_EVIDENCE_ORIGIN: dict[str, str] = {
    "policy_doc": "isms_policy_repository",
    "procedure": "document_center / SOP",
    "manual_review": "internal_audit_workbook",
    "training_record": "HRIS LMS export",
    "hr_export": "HRIS offboarding",
    "access_badge_log": "physical_access_system",
    "facility_checklist": "facility_inspection_app",
    "physical_audit": "cctv_sampling_log",
    "config_review": "change_mgmt / CMDB",
    "manual_test": "pen_test_or_checklist",
    "syslog_login": "ot/syslog · AAA",
    "syslog_cdp": "ot/syslog · CDP/LLDP",
    "syslog_ips": "ot/syslog · IPS/IDS",
    "syslog_config": "ot/syslog · CONFIG_I",
    "syslog_snmp": "ot/syslog · SNMP/crypto",
}


def _annex_index(annex_id: str) -> int:
    try:
        return int((annex_id or "A.0.0").split(".")[-1])
    except ValueError:
        return 0


def _etype_line(control: dict) -> str:
    return " · ".join(str(x) for x in (control.get("evidence_types") or ["manual_review"])[:4])


def build_evidence_sources(
    control: dict,
    *,
    tier: str,
    log_files: list[str] | None = None,
) -> list[dict]:
    annex_id = (control.get("annex_a_id") or "").strip()
    title = (control.get("title") or annex_id).strip()
    domain = control.get("domain") or "organizational"
    sources: list[dict] = []

    if tier == "auto_syslog":
        meta = _OT_MONITORED.get(annex_id, {})
        for et in meta.get("evidence_types") or control.get("evidence_types") or []:
            sources.append({
                "type": et,
                "origin": _EVIDENCE_ORIGIN.get(et, "syslog_classifier"),
            })
        sig = meta.get("wazuh_signal") or control.get("wazuh_signal")
        if sig:
            sources.append({"type": "wazuh_rule", "origin": sig})
        for f in (log_files or [])[:3]:
            sources.append({"type": "log_file", "origin": f})
        return sources

    idx = _annex_index(annex_id)
    domain_origin = {
        "organizational": [
            "GRC 政策庫",
            "法遵矩陣匯出",
            "董事會簽核包",
            "供應商合規入口",
            "風險登錄簿 API",
            "專案章程歸檔",
        ],
        "people": [
            "HRIS 訓練紀錄",
            "人資離職清單",
            "NDA 掃描批次",
            "到職查核表",
        ],
        "physical": [
            "門禁系統 CSV",
            "機房巡檢 App",
            "CCTV 抽樣表",
            "媒體銷毀簽收",
        ],
        "technological": [
            "CMDB 設定匯出",
            "弱掃報告摘要",
            "變更單系統",
            "備份還原紀錄",
            "憑證管理台",
        ],
    }
    primary_origin = domain_origin.get(domain, domain_origin["organizational"])
    pick = primary_origin[idx % len(primary_origin)]

    for et in control.get("evidence_types") or ["manual_review"]:
        sources.append({
            "type": et,
            "origin": _EVIDENCE_ORIGIN.get(et, pick),
        })
    sources.append({
        "type": "control_scope",
        "origin": f"{annex_id} · {title[:24]}",
    })
    return sources


def _syslog_semi_chain(annex_id: str, control: dict) -> list[dict]:
    meta = _OT_MONITORED.get(annex_id, {})
    key = control.get("control_key") or "ot_metric"
    sig = meta.get("wazuh_signal") or "regex bucket"
    profiles = {
        "A.5.15": (
            "SSH / RADIUS Syslog",
            "Core-SW · sec_login · AAA deny/accept",
            "access_control 分類桶",
            f"特權登入 · failed auth · {sig}",
        ),
        "A.5.19": (
            "鄰居與遠端維運 Syslog",
            "CDP/LLDP · remote_host · 供應商維護窗口",
            "supplier_security 分類",
            f"外部鄰居異常 · {sig}",
        ),
        "A.8.7": (
            "IPS / 惡意事件 Syslog",
            "malware · host_attack · USB 告警",
            "malware_defense 分類",
            f"IPS 命中 · {sig}",
        ),
        "A.8.8": (
            "IOS 版本 / SMU Syslog",
            "image · software · upgrade 事件",
            "patch_management 分類",
            f"版本漂移 · {sig}",
        ),
        "A.8.19": (
            "組態變更 Syslog",
            "CONFIG_I · interface · port-security",
            "recipe_audit 分類",
            f"未授權變更 · {sig}",
        ),
        "A.8.24": (
            "SNMP / PKI Syslog",
            "crypto · certificate · TLS session",
            "sec_gem_log 分類",
            f"密碼學使用 · {sig}",
        ),
    }
    p = profiles.get(annex_id, ("OT Syslog", "ot/ 設備", "分類桶", sig))
    title = control.get("title") or annex_id
    return [
        {"id": "s1", "label": p[0], "kind": "source", "detail": p[1]},
        {"id": "s2", "label": "Collector 解析", "kind": "process", "detail": f"Cisco 正規化 · {annex_id} {title}"},
        {"id": "s3", "label": p[2], "kind": "process", "detail": p[3]},
        {"id": "s4", "label": "Control Bundle", "kind": "artifact", "detail": f"{key} · 事件樣本 + KPI"},
        {"id": "s5", "label": "Evidence Registry", "kind": "evidence", "detail": f"evidence_id · SHA-256 · {annex_id}"},
        {"id": "s6", "label": "AI Auditor (LLM)", "kind": "process", "detail": f"引用 {key} bundle 原文 · Ollama"},
        {"id": "s7", "label": "Guardrail + AI Reviewer", "kind": "process", "detail": "護欄 · 控制項對映準確度"},
        {"id": "s8", "label": "Human Review Queue", "kind": "gate", "detail": f"{annex_id} review/fail 必核准"},
        {"id": "s9", "label": "合規宣稱", "kind": "claim", "detail": f"{title} · No Evidence, No Claim"},
    ]


def _syslog_traditional_chain(annex_id: str, control: dict) -> list[dict]:
    trad = {
        "A.5.15": [
            ("show logging · CLI 截圖", "逐台交換器登入紀錄"),
            ("Excel 登入矩陣", "帳號/時間人工整理"),
            ("Email 給 CISO", "附件無 hash"),
            ("PDF 存取控制章節", "稽核當下快照"),
            ("訪談確認 MFA", "與 log 弱關聯"),
        ],
        "A.5.19": [
            ("CDP 鄰居手動 show", "供應商 MAC 對照表"),
            ("Word 供應商清單", "版本分散"),
            ("Email 維護通知", "無集中索引"),
            ("PDF 供應商章節", "年度靜態"),
            ("訪談遠端維運", "口頭合規"),
        ],
        "A.8.7": [
            ("IPS 主控台截圖", "無 OT 集中 collector"),
            ("Excel 事件清單", "人工篩 malware"),
            ("Email SOC 摘要", "附件散落"),
            ("PDF 惡意軟體章", "難重算 KPI"),
            ("訪談防毒政策", "缺行級 trace"),
        ],
        "A.8.8": [
            ("show version 截圖", "逐台 IOS 盤點"),
            ("Excel 修補矩陣", "SMU 狀態人工"),
            ("Email 修補計畫", "無密碼學校驗"),
            ("PDF 弱點管理", "季度快照"),
            ("訪談修補窗口", "與 syslog 脫鉤"),
        ],
        "A.8.19": [
            ("running-config 備份", "工程師手動 diff"),
            ("Word 變更紀錄", "recipe 版本分散"),
            ("Email 變更核准", "無 evidence_id"),
            ("PDF 組態稽核", "難重播變更"),
            ("訪談變更流程", "缺乏自動分類"),
        ],
        "A.8.24": [
            ("SNMP 社群字串查核", "CLI 逐台"),
            ("Excel 憑證到期表", "人工提醒"),
            ("Email PKI 通知", "無 hash chain"),
            ("PDF 密碼學章節", "靜態合規"),
            ("訪談金鑰管理", "與 log 弱連結"),
        ],
    }
    steps = trad.get(annex_id)
    if not steps:
        return _generic_traditional(control, "technological")
    return [
        {"id": f"t{i + 1}", "label": lab, "kind": "source" if i == 0 else "process", "detail": det}
        for i, (lab, det) in enumerate(steps)
    ]


def _import_semi_chain(control: dict) -> list[dict]:
    annex_id = (control.get("annex_a_id") or "").strip()
    title = (control.get("title") or annex_id).strip()
    domain = control.get("domain") or "organizational"
    idx = _annex_index(annex_id)
    et = _etype_line(control)

    ingest_labels = {
        "organizational": [
            ("政策庫批次", "GRC zip · 簽核 PDF"),
            ("程序書匯入", "SOP 版本樹"),
            ("風險與例外", "risk_register.json"),
            ("供應商包", "vendor_compliance_bundle"),
            ("專案安全檢查", "project_security_gate"),
        ],
        "people": [
            ("HR 訓練匯出", "LMS completion CSV"),
            ("離職/調職", "HRIS termination feed"),
            ("保密協議", "NDA scanned batch"),
            ("背景調查", "screening checklist"),
        ],
        "physical": [
            ("門禁紀錄", "badge_swipe export"),
            ("機房巡檢", "inspection photos + form"),
            ("CCTV 抽樣", "retention audit sample"),
            ("媒體銷毀", "destruction certificate"),
        ],
        "technological": [
            ("設定/CMDB", "config baseline export"),
            ("掃描摘要", "vuln scan digest"),
            ("變更紀錄", "change ticket API"),
            ("備份證明", "restore test log"),
            ("金鑰/憑證", "cert inventory"),
        ],
    }
    variants = ingest_labels.get(domain, ingest_labels["organizational"])
    lab, det = variants[idx % len(variants)]

    parse_detail = {
        "organizational": f"OCR+結構化 · 《{title}》→ {annex_id}",
        "people": f"HR 欄位對照 · {title} · training/HR 對映",
        "physical": f"巡檢表 OCR · {title} · 實體證據索引",
        "technological": f"設定 diff · {title} · config_review 對映",
    }.get(domain, f"Annex 對映 · {annex_id}")

    check_variants = [
        f"必填欄位 · 版本 · 期間 · {et}",
        f"簽核鏈 · owner · 涵蓋 Fab OT · {annex_id}",
        f"SoD 抽樣 · 例外清單 · {title[:16]}",
    ]
    check_detail = check_variants[idx % len(check_variants)]

    return [
        {"id": "s1", "label": lab, "kind": "source", "detail": f"{det} · {annex_id}"},
        {"id": "s2", "label": "AI 文件解析", "kind": "process", "detail": parse_detail},
        {"id": "s3", "label": "完整性檢查", "kind": "process", "detail": check_detail},
        {"id": "s4", "label": "Control Bundle", "kind": "artifact", "detail": f"{annex_id} 摘要 · metric · {et}"},
        {"id": "s5", "label": "Evidence Registry", "kind": "evidence", "detail": f"hash chain · mi-* / import batch · {annex_id}"},
        {"id": "s6", "label": "AI Auditor (LLM)", "kind": "process", "detail": f"引用《{title}》匯入原文 · 禁虛構"},
        {"id": "s7", "label": "Guardrail + AI Reviewer", "kind": "process", "detail": f"對映準確度 · {domain} 規則集"},
        {"id": "s8", "label": "Human Review Queue", "kind": "gate", "detail": f"{annex_id} 文件證據 review 閘門"},
        {"id": "s9", "label": "合規宣稱", "kind": "claim", "detail": f"{title} · evidence 綁定 {annex_id}"},
    ]


def _generic_traditional(control: dict, domain: str | None = None) -> list[dict]:
    domain = domain or control.get("domain") or "organizational"
    title = (control.get("title") or control.get("annex_a_id") or "").strip()
    annex_id = (control.get("annex_a_id") or "").strip()
    idx = _annex_index(annex_id)
    templates = {
        "organizational": [
            [("政策 Word/PDF", "法遵部門手動整理"), ("Excel 控制矩陣", "Annex 對照人工"), ("Email 簽核", "無 hash"), ("PDF ISMS 包", "年度快照"), ("管理審查訪談", "口頭宣稱")],
            [("GRC 匯出 USB", "離線傳遞"), ("試算表 RACI", "角色責任人工"), ("會議紀錄 Email", "版本分散"), ("靜態稽核報告", "難重算 KPI"), ("文件缺失訪談", "補證說明")],
        ],
        "people": [
            [("HR 紙本訓練簽到", "掃描後 Email"), ("Excel 到離職", "人工對 HRIS"), ("NDA 紙本", "無集中索引"), ("PDF 人員章", "稽核快照"), ("訪談保密", "弱 trace")],
        ],
        "physical": [
            [("門禁紙本登記", "警衛台帳"), ("Excel 巡檢", "照片散落"), ("Email 機房鑰匙", "無校驗"), ("PDF 實體安全", "靜態"), ("現場 walkthrough", "口頭合規")],
        ],
        "technological": [
            [("CLI 設定截圖", "逐台無 collector"), ("Excel 資產表", "CMDB 脫節"), ("Email 變更單", "附件 hash 缺"), ("PDF 技術章", "難重播"), ("訪談組態", "與 log 脫鉤")],
            [("手動 show run", "備份分散"), ("Word 測試紀錄", "manual_test"), ("工單 Email", "無 evidence_id"), ("稽核 PDF", "季度靜態"), ("訪談備份", "口頭宣稱")],
        ],
    }
    pool = templates.get(domain, templates["organizational"])
    row = pool[idx % len(pool)]
    kinds = ["source", "artifact", "process", "artifact", "claim"]
    return [
        {"id": f"t{i + 1}", "label": lab, "kind": kinds[i], "detail": f"{det} · {title[:20]}"}
        for i, (lab, det) in enumerate(row)
    ]


def build_semi_shield_chain(control: dict, tier: str) -> list[dict]:
    annex_id = (control.get("annex_a_id") or "").strip()
    if tier == "auto_syslog":
        return _syslog_semi_chain(annex_id, control)
    return _import_semi_chain(control)


def build_traditional_chain(control: dict, tier: str) -> list[dict]:
    annex_id = (control.get("annex_a_id") or "").strip()
    if tier == "auto_syslog":
        return _syslog_traditional_chain(annex_id, control)
    return _generic_traditional(control)
