"""
生成輸出幻覺偵測訓練資料。
label: 0 = grounded（有依據／知識正確）, 1 = hallucinated（捏造／背離證據）

每筆格式：
  question, evidence, answer, label, category
推論時會組成：[Q] ... [E] ... [A] ...
"""
from __future__ import annotations

import json
from pathlib import Path

OUTPUT_PATH = Path(__file__).with_name("hallucination_dataset.json")

# --- 共用證據模板 ---
EVIDENCE_COMPLIANCE = (
    "【監控摘要】access_control: fail count=12; malware_defense: review count=3; "
    "recipe_audit: pass count=0; sec_gem_log: fail count=5; "
    "patch_management: review count=2; supplier_security: pass count=0"
)
EVIDENCE_LOGIN = (
    "%SEC_LOGIN-4-LOGIN_FAILED: Login failed [user: admin] [source: 192.168.3.10] "
    "[localport: 22] [Reason: Login Authentication Failed]"
)
EVIDENCE_UPDOWN = (
    "%LINK-3-UPDOWN: Interface GigabitEthernet1/0/24, changed state to down "
    "then up; %LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet1/0/24"
)
EVIDENCE_SNMP = (
    "%SNMP-3-AUTHFAIL: Authentication failure for SNMP request from host 10.1.2.3 "
    "community public"
)
EVIDENCE_EMPTY = "（無可用日誌或監控資料）"

dataset: list[dict] = []

# ========== GROUNDED (0) ==========

def _add(q, e, a, cat):
    dataset.append({"question": q, "evidence": e, "answer": a, "label": 0, "category": cat})

# 合規現況 — 依監控數字
_add(
    "目前合規現況如何？",
    EVIDENCE_COMPLIANCE,
    "依監控掃描：存取控制 fail（12 筆）、惡意防護 review（3 筆）、"
    "配方稽核 pass、SEC/GEM 日誌 fail（5 筆）、修補管理 review（2 筆）、供應商安全 pass。",
    "grounded_compliance",
)
_add(
    "各控制項事件量？",
    EVIDENCE_COMPLIANCE,
    "access_control 12 筆 fail、malware_defense 3 筆 review、sec_gem_log 5 筆 fail。",
    "grounded_metrics",
)

# 登入失敗 — 依 syslog
_add(
    "分析這段登入日誌",
    EVIDENCE_LOGIN,
    "SEC_LOGIN-4-LOGIN_FAILED 表示來源 192.168.3.10 對 admin 帳號 SSH 登入失敗，"
    "建議對帳來源 IP 與帳號政策（A.7.4／A.11.2）。",
    "grounded_syslog",
)
_add(
    "這則 Cisco syslog 代表什麼？",
    EVIDENCE_LOGIN,
    "這是登入驗證失敗事件，來源 192.168.3.10、使用者 admin、本端 port 22。",
    "grounded_syslog",
)

# UPDOWN — 依證據
_add(
    "介面 UPDOWN 事件怎麼看？",
    EVIDENCE_UPDOWN,
    "GigabitEthernet1/0/24 發生 link down/up 與 line protocol 變更，"
    "需排查實體連線或 errdisable（A.8.19）。",
    "grounded_syslog",
)

# SNMP — 依證據
_add(
    "SNMP 認證失敗代表什麼？",
    EVIDENCE_SNMP,
    "SNMP-3-AUTHFAIL 表示來源 10.1.2.3 使用 community public 認證失敗，"
    "應改 SNMPv3 並汰換預設 community（A.8.24）。",
    "grounded_syslog",
)

# ISO 27001 知識 — 無特定 evidence 但回答正確
_iso_qa = [
    (
        "什麼是 ISO 27001？",
        "ISO/IEC 27001 是資訊安全管理系統（ISMS）要求標準，"
        "定義建立、實施、維運與改善 ISMS 的管理要求，Annex A 列出控制參考。",
    ),
    (
        "ISO 27001 和 27002 差異？",
        "27001 是 ISMS 認證要求標準；27002 提供 Annex A 控制措施的實務指引，"
        "27001 要求組織選用並實施控制。",
    ),
    (
        "27001 包含哪些項目？",
        "27001 包含管理體系條款（領導、規劃、支援、營運、績效、改善）"
        "與 Annex A 資訊安全控制參考。",
    ),
]
for q, a in _iso_qa:
    _add(q, EVIDENCE_EMPTY, a, "grounded_knowledge")

# 加固步驟 — 通用正確知識
_add(
    "如何停用 Telnet 改用 SSH？",
    EVIDENCE_EMPTY,
    "在 Cisco 交換器執行：line vty 0 15 → transport input ssh → "
    "no transport input telnet；並確認 AAA／RADIUS 已設定（A.8.24）。",
    "grounded_howto",
)
_add(
    "修補步驟有哪些？",
    EVIDENCE_COMPLIANCE,
    "存取控制 fail：盤點特權帳、強制 SSH；SEC/GEM fail：改 SNMPv3、限制 NMS 來源；"
    "修補管理 review：排程維護窗升級韌體。",
    "grounded_remediation",
)

# 寒暄 / 短答
_add("你好", EVIDENCE_EMPTY, "您好，我是 Semi-Shield Cyber Agent，可協助 OT 合規與 ISO 27001 相關問題。", "grounded_casual")
_add("謝謝", EVIDENCE_EMPTY, "不客氣，如需合規現況、修補步驟或 syslog 分析，歡迎再問。", "grounded_casual")

# 變體：同一 evidence 多種正確說法
for n in (12, 12, 12):
    _add(
        "存取控制狀態？",
        EVIDENCE_COMPLIANCE,
        f"access_control 目前 fail，共 {n} 筆事件。",
        "grounded_metrics",
    )

# ========== HALLUCINATED (1) ==========

def _add_bad(q, e, a, cat):
    dataset.append({"question": q, "evidence": e, "answer": a, "label": 1, "category": cat})

# 捏造設備／機台
_add_bad(
    "合規現況如何？",
    EVIDENCE_COMPLIANCE,
    "MFG01 控制器遭未授權存取，P01 機臺配方被篡改，建議立即隔離產線。",
    "halluc_device",
)
_add_bad(
    "分析監控現況",
    EVIDENCE_COMPLIANCE,
    "MFG02 產線 PLC 發生 Modbus 寫入異常，已開啟閥門造成製程中斷。",
    "halluc_device",
)

# 捏造 syslog
_add_bad(
    "登入日誌分析",
    EVIDENCE_LOGIN,
    "[INFO] A.8.7 本次日誌：2024-03-15 HOSTNAME 發生重放攻擊並成功開啟 Modbus 閥門。",
    "halluc_syslog",
)
_add_bad(
    "這段日誌什麼意思？",
    EVIDENCE_LOGIN,
    "SYSLOG: [ID 1000] 192.168.1.100 對 [HOSTNAME] 進行配方下載與控制器變更組態。",
    "halluc_syslog",
)

# 假 IP／假時間
_add_bad(
    "合規診斷",
    EVIDENCE_COMPLIANCE,
    "2023-08-01 Admin 使用者提交報告工單 #12345，預計 2 小時內完成，事件狀態：執行中。",
    "halluc_fake_status",
)
_add_bad(
    "風險分析",
    EVIDENCE_EMPTY,
    "設備 192.168.1.100 於 2024-06-01 遭重放攻擊，SYSLOG: [ID 1000] 密碼學與網絡傳輸安全校驗檢查結果異常。",
    "halluc_fake_event",
)

# 錯誤計數
_add_bad(
    "各控制項筆數？",
    EVIDENCE_COMPLIANCE,
    "access_control 12345 筆、malware_defense 45678 筆、network_monitor 999 筆 fail。",
    "halluc_metrics",
)
_add_bad(
    "合規現況",
    EVIDENCE_COMPLIANCE,
    "六項控制全部 pass，無任何 fail 或 review 事件。",
    "halluc_metrics",
)

# 錯誤 ISO 編號
_add_bad(
    "什麼是 ISO 27001？",
    EVIDENCE_EMPTY,
    "ISO 22007 是資訊安全管理系統要求，包含 ISO 25701 與 27101 控制域。",
    "halluc_iso",
)
_add_bad(
    "ISO 27001 說明",
    EVIDENCE_EMPTY,
    "Security Techniques — Requirements 標準 ISO/IEC 29001 系列定義 ISMS。",
    "halluc_iso",
)

# 異場域
_add_bad(
    "OT 合規現況",
    EVIDENCE_COMPLIANCE,
    "汽車組裝廠 PLC 與食品廠 SCADA 均發現 Telnet 明文登入，需立即加固。",
    "halluc_foreign_site",
)

# UPDOWN 證據卻講重放／閥門
_add_bad(
    "介面事件分析",
    EVIDENCE_UPDOWN,
    "偵測到重放攻擊 Replay Attack，攻擊者透過 Modbus 寫入開啟閥門並下載配方。",
    "halluc_wrong_topic",
)

# 證據只有監控摘要卻捏造具體 syslog 行
_add_bad(
    "合規現況摘要",
    EVIDENCE_COMPLIANCE,
    "本次日誌：%MALWARE-4-MALWARE_DETECTED on GigabitEthernet1/0/1 from 10.9.8.7",
    "halluc_invented_syslog",
)

# 訓練殘留／提示詞洩漏
_add_bad(
    "合規分析",
    EVIDENCE_COMPLIANCE,
    "【防幻覺鐵律】禁止捏造。## 地端 LLM 智慧合規診斷報告 一、事件經過摘要",
    "halluc_prompt_leak",
)
_add_bad(
    "你好",
    EVIDENCE_EMPTY,
    "使用者：你好 assistant：請用繁體中文 model",
    "halluc_train_leak",
)

# 英文幻覺牆
_add_bad(
    "ISO 27001 是什麼？",
    EVIDENCE_EMPTY,
    "Security Techniques — Requirements and Code of Practice for Information Security Management Systems",
    "halluc_english",
)

# 擴增：複製 grounded 並改一個關鍵字變 hallucination
_grounded_copy = [d for d in dataset if d["label"] == 0]
for item in _grounded_copy[:20]:
    if "12" in item["answer"]:
        bad = dict(item)
        bad["answer"] = item["answer"].replace("12", "99999")
        bad["label"] = 1
        bad["category"] = "halluc_aug_metrics"
        dataset.append(bad)

# 平衡：再補一些 grounded 變體
_extra_grounded = [
    ("SNMP 問題怎麼修？", EVIDENCE_SNMP, "改 SNMPv3、移除 public community、限制 NMS 來源 IP。", "grounded_remediation"),
    ("access_control fail 怎麼辦？", EVIDENCE_COMPLIANCE, "盤點共用帳、改個人帳與最小權限、強制 SSH。", "grounded_remediation"),
    ("malware_defense review 代表？", EVIDENCE_COMPLIANCE, "惡意防護控制項為 review，共 3 筆待複核。", "grounded_metrics"),
    ("配方稽核狀態？", EVIDENCE_COMPLIANCE, "recipe_audit 目前 pass，count=0。", "grounded_metrics"),
    ("貼 %SEC_LOGIN 日誌", EVIDENCE_LOGIN, "登入失敗事件，需檢查來源 IP 與帳號鎖定政策。", "grounded_syslog"),
]
for q, e, a, c in _extra_grounded:
    _add(q, e, a, c)

# 再補 hallucination 變體
_extra_bad = [
    ("修補建議", EVIDENCE_COMPLIANCE, "刪除所有日誌後重裝系統即可恢復合規。", "halluc_bad_advice"),
    ("合規嗎？", EVIDENCE_COMPLIANCE, "全部控制項均已通過 ISO 9001 認證，無需改善。", "halluc_wrong_std"),
    ("日誌分析", EVIDENCE_EMPTY, "%CONFIG-6-UPDATE: 新增新設定檔並刪除舊設定檔，MFG01 控制器進行變更組態。", "halluc_no_evidence"),
]
for q, e, a, c in _extra_bad:
    _add_bad(q, e, a, c)

# --- 批量擴增： grounded / hallucinated 模板 ---
_CONTROLS = [
    ("access_control", "fail", 12),
    ("malware_defense", "review", 3),
    ("recipe_audit", "pass", 0),
    ("sec_gem_log", "fail", 5),
    ("patch_management", "review", 2),
    ("supplier_security", "pass", 0),
]
for key, status, cnt in _CONTROLS:
    ev = f"【監控】{key}: {status} count={cnt}"
    _add(f"{key} 狀態？", ev, f"{key} 為 {status}，共 {cnt} 筆。", "grounded_aug")
    _add_bad(
        f"{key} 狀態？",
        ev,
        f"{key} 全部 pass，共 0 筆 fail。",
        "halluc_aug_status",
    )
    _add_bad(
        f"{key} 分析",
        ev,
        f"network_monitor 對 {key} 偵測到 88888 筆異常。",
        "halluc_aug_fake_ctrl",
    )

_SYSLOG_TEMPLATES = [
    (
        "%SEC_LOGIN-4-LOGIN_FAILED: Login failed [user: {user}] [source: {ip}]",
        "登入失敗，使用者 {user}，來源 {ip}。",
    ),
    (
        "%LINK-3-UPDOWN: Interface GigabitEthernet1/0/{port}, changed state to down",
        "介面 GigabitEthernet1/0/{port} link down，需排查連線。",
    ),
    (
        "%SNMP-3-AUTHFAIL: Authentication failure from host {ip} community {comm}",
        "SNMP 認證失敗，來源 {ip}，community {comm}。",
    ),
]
for tpl, ans_tpl in _SYSLOG_TEMPLATES:
    for i in range(3):
        ip = f"192.168.{i+1}.{10+i}"
        ev = tpl.format(user="admin", ip=ip, port=10 + i, comm="public")
        ans = ans_tpl.format(user="admin", ip=ip, port=10 + i, comm="public")
        _add("分析 syslog", ev, ans, "grounded_aug_syslog")
        _add_bad(
            "分析 syslog",
            ev,
            f"MFG01 控制器 {ip} 遭 Modbus 重放攻擊開啟閥門。",
            "halluc_aug_syslog",
        )

_HALLUC_PATTERNS = [
    "2023-08-01 Admin 提交工單 #12345，報告狀態執行中。",
    "SYSLOG: [ID 1000] [HOSTNAME] 192.168.1.100 配方下載。",
    "ISO 22007 與 25701 定義 ISMS 要求。",
    "汽車組裝廠與食品廠 OT 均需加固。",
    "【防幻覺鐵律】## 地端 LLM 智慧合規診斷報告",
    "Security Techniques — Requirements for ISMS",
]
for i, bad_ans in enumerate(_HALLUC_PATTERNS):
    _add_bad(f"合規問題 {i}", EVIDENCE_COMPLIANCE, bad_ans, "halluc_aug_pattern")

_ISO_GOOD = [
    ("27001 是什麼", "ISMS 要求標準，含管理體系條款與 Annex A。"),
    ("27002 用途", "提供 Annex A 控制措施實務指引。"),
    ("ISMS 定義", "資訊安全管理系統，用 PDCA 持續改善。"),
]
for q, a in _ISO_GOOD:
    _add(q, EVIDENCE_EMPTY, a, "grounded_aug_iso")


def main():
    out = OUTPUT_PATH
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    n0 = sum(1 for d in dataset if d["label"] == 0)
    n1 = sum(1 for d in dataset if d["label"] == 1)
    print(f"已寫入 {out}")
    print(f"  grounded (0): {n0}")
    print(f"  hallucinated (1): {n1}")
    print(f"  total: {len(dataset)}")


if __name__ == "__main__":
    main()
