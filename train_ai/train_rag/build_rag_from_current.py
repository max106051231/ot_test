"""
把「目前專案資料」做成 RAG 資料集（半導體廠／Cisco syslog／ISO 27001）。

來源：
  1) train_ai/train_llm/train.json          → 既有 Q&A
  2) ot/*.txt 設備日誌（依訊息類型抽樣） → 正確對應控制項的診斷知識
  3) 舊 knowledge_base.json（可選）       → 僅保留半導體／通用工控，剔除異場域

產出：
  - knowledge_base.json
  - rag_pairs.jsonl
  - knowledge_base.backup_*.json（覆寫前備份）

用法：
  cd train_ai/train_rag
  python build_rag_from_current.py                      # 預設：只做 log 分析知識
  python build_rag_from_current.py --include-train      # 另加 train.json 理論 Q&A
  python build_rag_from_current.py --include-old-kb     # 另加過濾後舊 KB
  python build_rag_from_current.py --rebuild-index
  python build_rag_from_current.py --per-type 8
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN_AI = ROOT.parent
PROJECT = TRAIN_AI.parent
OT_DIR = PROJECT / "ot"
TRAIN_JSON = TRAIN_AI / "train_llm" / "train.json"
KB_PATH = ROOT / "knowledge_base.json"
PAIRS_PATH = ROOT / "rag_pairs.jsonl"

SITE = "半導體廠"
FOREIGN_RE = re.compile(
    r"汽車組裝廠|汽車廠|車廠|食品廠|鋼鐵廠|石化廠|水泥廠|紙漿廠|"
    r"天然氣調壓站|水廠|港口碼頭|風力發電場|火力電廠|充電場站|"
    r"資料中心電力|醫院機電|製藥廠|面板廠|機場助航|BESS\s*場站|變電所|捷運",
    re.I,
)
CISCO_RE = re.compile(r"%([A-Z0-9_-]+)-(\d+)-([A-Z0-9_]+):\s*(.*)$")

CONTROL_META = {
    "access_control": {
        "iso": "A.7.4 / A.11.2",
        "title": "實體安全與存取邊界／身分驗證",
    },
    "sec_gem_log": {
        "iso": "A.8.24",
        "title": "密碼學與網絡傳輸安全",
    },
    "patch_management": {
        "iso": "A.8.8",
        "title": "技術弱點管理",
    },
    "supplier_security": {
        "iso": "A.5.19",
        "title": "供應商資安關係",
    },
    "malware_defense": {
        "iso": "A.8.7",
        "title": "惡意軟體防禦",
    },
    "recipe_audit": {
        "iso": "A.8.19",
        "title": "組態變更管理稽核",
    },
}

# facility/mnemonic → 控制項（與監控分類對齊，避免 LOGIN 被誤判成重放攻擊）
TYPE_RULES = [
    ("access_control", re.compile(
        r"SEC_LOGIN|LOGIN|LOGOUT|AUTH|AAA|RADIUS|TACACS|SSH|TELNET|"
        r"DOT1X|MAB_|USER_LOCKED|TTY_EXPIRE|LOCAL_LOGIN",
        re.I,
    )),
    ("sec_gem_log", re.compile(
        r"SNMP|CRYPTO|PKI|TLS|SSL|IPSEC|IKE|CERTIFICATE|SUDI|COMMUNITY",
        re.I,
    )),
    ("patch_management", re.compile(
        r"BOOT|IMAGE|VERSION|PLATFORM|SOFTWARE|UPGRADE|IOSXE|INSTALL|SMU|FW_",
        re.I,
    )),
    ("supplier_security", re.compile(
        r"CDP|LLDP|NEIGHBOR|LOGGINGHOST|PNP_|BGP|TRAP|REMOTE",
        re.I,
    )),
    ("malware_defense", re.compile(
        r"IPS|IDS|MALWARE|VIRUS|THREAT|PORTSCAN|DOS|DDOS|USB_DEVICE|"
        r"FW-3-MALWARE|HOST_ATTACK|PSECURE",
        re.I,
    )),
    ("recipe_audit", re.compile(
        r"CONFIG|SYS-|LINK-|LINEPROTO|ILPOWER|PARSER|SPANTREE|"
        r"ERR_DISABLE|UPDOWN|POWER_|ENV-|FAN_|TEMP_|RELOAD|CMD_DENIED",
        re.I,
    )),
]


def classify_key(facility: str, mnemonic: str, body: str) -> str:
    hay = f"{facility} {mnemonic} {body}"
    for key, pat in TYPE_RULES:
        if pat.search(hay):
            return key
    return "recipe_audit"


def sanitize(text: str) -> str:
    if not text:
        return text
    t = FOREIGN_RE.sub(SITE, text)
    t = re.sub(r"\**場域情境（[^）)]{1,40}）\**[：:][^\n]*", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def make_doc(doc_id: str, instruction: str, output: str, source: str, tags=None) -> dict:
    instruction = sanitize(instruction)
    output = sanitize(output)
    return {
        "id": doc_id,
        "title": instruction[:120],
        "instruction": instruction,
        "output": output,
        "content": output,
        "text": f"問題: {instruction}\n答案: {output}",
        "source": source,
        "tags": tags or [],
        "domain": SITE,
    }


def load_train_json() -> list[dict]:
    if not TRAIN_JSON.is_file():
        print(f"⚠️ 找不到 {TRAIN_JSON}，略過 train.json")
        return []
    rows = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
    docs = []
    for i, row in enumerate(rows):
        q = (row.get("instruction") or "").strip()
        a = (row.get("output") or "").strip()
        if not q or not a:
            continue
        # 跳過明顯異場域且無半導體／Cisco／ISO 關鍵字的舊模板
        blob = q + a
        if FOREIGN_RE.search(blob) and not re.search(
            r"半導體|Cisco|ISO\s*27001|C9300|SEC_LOGIN|RADIUS|Modbus|PLC", blob, re.I
        ):
            continue
        docs.append(
            make_doc(f"train_{i:04d}", q, a, "train_json", tags=["train_json"])
        )
    print(f"✓ train.json → {len(docs)} 篇")
    return docs


def analysis_for_event(facility: str, severity: str, mnemonic: str, body: str, raw: str) -> tuple[str, str, str]:
    """回傳 (control_key, instruction, output)。"""
    key = classify_key(facility, mnemonic, body)
    meta = CONTROL_META[key]
    code = f"%{facility}-{severity}-{mnemonic}"
    hay = f"{facility} {mnemonic} {body}".upper()

    # —— 依訊息類型給「正確」診斷（避免 LOGIN → 重放攻擊）——
    if "LOGIN_SUCCESS" in hay or mnemonic == "LOGIN_SUCCESS":
        port_note = ""
        if re.search(r"localport:\s*23\b", body, re.I) or "TELNET" in hay:
            port_note = (
                "偵測到 **Telnet（port 23）** 登入：屬明文管理通道，"
                "在半導體廠 OT 管理面屬高風險，應對應關閉並改強制 SSH。"
            )
        elif re.search(r"localport:\s*22\b", body, re.I) or "SSH" in hay:
            port_note = "管理通道為 **SSH（port 22）**，方向正確，仍須確認來源 IP 白名單與個人帳號。"
        user_m = re.search(r"user:\s*([^\]]+)", body, re.I)
        src_m = re.search(r"Source:\s*([\d.]+)", body, re.I)
        user = (user_m.group(1).strip() if user_m else "（未知）")
        src = (src_m.group(1) if src_m else "（未知）")
        instruction = (
            f"請依下列 Cisco syslog 做 ISO 27001 合規診斷（{SITE}）：\n{raw.strip()}"
        )
        output = (
            f"## 地端 LLM 智慧合規診斷報告\n\n"
            f"## 一、事件經過摘要\n"
            f"設備記錄 `{code}`：帳號 `{user}` 自來源 `{src}` 登入成功。"
            f"此為**管理面身分驗證／遠端存取事件**，不是現場 PLC 控制指令，也不是重放攻擊。\n"
            f"{port_note}\n\n"
            f"## 二、不合規／風險分析\n"
            f"- 控制項：{meta['iso']} {meta['title']}\n"
            f"- 若使用共用帳（如 cisco），無法個人究責，違反最小權限與帳號管理要求。\n"
            f"- 若來源 IP 非網管／跳板白名單，存在未授權遠端維運風險。\n"
            f"- **禁止**將本類日誌解讀為 ICS 指令重放、Modbus 寫入或現場閥門操控。\n\n"
            f"## 三、具體修補建議\n"
            f"- 禁用 Telnet，僅允許 SSHv2；管理 ACL 限制來源網段。\n"
            f"- 改 AAA（RADIUS/TACACS+）個人帳，廢止共用帳。\n"
            f"- 對非白名單 `LOGIN_SUCCESS` 做 SIEM 告警並對帳變更單。\n"
            f"- 啟用閒置逾時（TTY exec-timeout）並保留完整登入／登出軌跡。\n"
        )
        return key, instruction, output

    if "LOGIN_FAILED" in hay:
        instruction = f"請分析此登入失敗日誌並對應 ISO 控制項（{SITE}）：\n{raw.strip()}"
        output = (
            f"## 地端 LLM 智慧合規診斷報告\n\n"
            f"## 一、事件經過摘要\n"
            f"`{code}` 顯示管理登入失敗。可能為密碼錯誤、帳號鎖定前兆或探測行為。\n\n"
            f"## 二、不合規／風險分析\n"
            f"- 控制項：{meta['iso']} {meta['title']}\n"
            f"- 短時間大量失敗可能為暴力破解或憑證噴灑。\n"
            f"- 需與後續 `USER_LOCKED`／來源 IP 關聯分析。\n\n"
            f"## 三、具體修補建議\n"
            f"- 啟用登入失敗鎖定與來源限速；僅白名單可達管理面。\n"
            f"- 改多因子／金鑰登入；監控失敗突增告警。\n"
            f"- 確認是否外包商或錯誤跳板，留下事件單。\n"
        )
        return key, instruction, output

    if "LOGOUT" in hay or "TTY_EXPIRE" in hay:
        instruction = f"請解釋此連線結束／逾時日誌的合規意義（{SITE}）：\n{raw.strip()}"
        output = (
            f"## 地端 LLM 智慧合規診斷報告\n\n"
            f"## 一、事件經過摘要\n"
            f"`{code}` 表示管理 session 正常結束或因閒置逾時被中止，屬連線生命週期事件。\n\n"
            f"## 二、不合規／風險分析\n"
            f"- 控制項：{meta['iso']} {meta['title']}\n"
            f"- 有逾時機制通常是正面控制；若缺少對應 LOGIN 則稽核鏈不完整。\n"
            f"- 仍須確認登入帳號是否為個人帳、來源是否合法。\n\n"
            f"## 三、具體修補建議\n"
            f"- 維持合理 exec-timeout；集中保存 LOGIN／LOGOUT 供稽核。\n"
            f"- 與變更單／維運時段對帳，異常時段登出需追查。\n"
        )
        return key, instruction, output

    if "SNMP" in hay or "AUTHFAIL" in hay or "COMMUNITY" in hay:
        instruction = f"請診斷此 SNMP／傳輸安全相關日誌（{SITE}）：\n{raw.strip()}"
        output = (
            f"## 地端 LLM 智慧合規診斷報告\n\n"
            f"## 一、事件經過摘要\n"
            f"`{code}` 與 SNMP／認證失敗或傳輸安全有關，常見於錯誤 community 或未授權輪詢。\n\n"
            f"## 二、不合規／風險分析\n"
            f"- 控制項：{meta['iso']} {meta['title']}\n"
            f"- SNMPv1/v2c 明文 community 易遭竊聽與濫用。\n"
            f"- 未授權查詢可能被用於資產探測。\n\n"
            f"## 三、具體修補建議\n"
            f"- 升級 SNMPv3（authPriv）；輪替 community；ACL 限制 NMS 來源。\n"
            f"- 關閉不必要 SNMP；監控 AUTHFAIL 暴增。\n"
        )
        return key, instruction, output

    if re.search(r"CRYPTO|IKE|PKI|CERTIFICATE|SSH2|KEY_SIZE", hay):
        instruction = f"請分析此加密／憑證／SSH 相關事件（{SITE}）：\n{raw.strip()}"
        output = (
            f"## 地端 LLM 智慧合規診斷報告\n\n"
            f"## 一、事件經過摘要\n"
            f"`{code}` 涉及金鑰交換、憑證或 SSH 參數，屬於傳輸與密碼學控制範圍。\n\n"
            f"## 二、不合規／風險分析\n"
            f"- 控制項：{meta['iso']} {meta['title']}\n"
            f"- 憑證無效、金鑰過短或 IKE 失敗會削弱管理面與隧道安全。\n\n"
            f"## 三、具體修補建議\n"
            f"- 更新憑證鏈、禁用弱密碼套件；SSH 使用足夠金鑰長度。\n"
            f"- 定期檢查 CRYPTO/PKI 告警並納入變更窗口處理。\n"
        )
        return key, instruction, output

    if re.search(r"MALWARE|IPS-|IDS-|SIGNATURE|USB_DEVICE|PSECURE", hay):
        instruction = f"請診斷此惡意／入侵／埠安全事件（{SITE}）：\n{raw.strip()}"
        output = (
            f"## 地端 LLM 智慧合規診斷報告\n\n"
            f"## 一、事件經過摘要\n"
            f"`{code}` 顯示惡意軟體攔截、IPS 簽章或埠安全違規，需視為資安事件處理。\n\n"
            f"## 二、不合規／風險分析\n"
            f"- 控制項：{meta['iso']} {meta['title']}\n"
            f"- 可能影響無塵室網段可用性或成為橫向移動跳板。\n\n"
            f"## 三、具體修補建議\n"
            f"- 隔離來源埠／主機；保全證據；更新簽章與白名單。\n"
            f"- 禁用未授權 USB；強化 port-security 與 802.1X。\n"
        )
        return key, instruction, output

    if re.search(r"CONFIG_I|PARSER|CMD_DENIED|RELOAD", hay):
        instruction = f"請依組態變更稽核角度分析（{SITE}）：\n{raw.strip()}"
        output = (
            f"## 地端 LLM 智慧合規診斷報告\n\n"
            f"## 一、事件經過摘要\n"
            f"`{code}` 屬組態變更、指令拒絕或設備重載相關 syslog。\n\n"
            f"## 二、不合規／風險分析\n"
            f"- 控制項：{meta['iso']} {meta['title']}\n"
            f"- 無變更單的 CONFIG 變更可能破壞基線與分段。\n\n"
            f"## 三、具體修補建議\n"
            f"- 強制變更單號；archive/config 差異比對；限制可寫入管理帳。\n"
            f"- 重載需維護窗口與回退映像準備。\n"
        )
        return key, instruction, output

    if re.search(r"CDP|LLDP|PNP_|LOGGINGHOST|NEIGHBOR", hay):
        instruction = f"請評估此外部發現／遠端日誌／供應商通道事件（{SITE}）：\n{raw.strip()}"
        output = (
            f"## 地端 LLM 智慧合規診斷報告\n\n"
            f"## 一、事件經過摘要\n"
            f"`{code}` 涉及鄰近裝置發現、PnP 或遠端 logging 等外部／供應商相關通道。\n\n"
            f"## 二、不合規／風險分析\n"
            f"- 控制項：{meta['iso']} {meta['title']}\n"
            f"- 未受控的 CDP/LLDP/PnP 可能洩漏拓樸或引入未授權裝置。\n\n"
            f"## 三、具體修補建議\n"
            f"- 非必要關閉 CDP/LLDP；遠端 log 僅送信任 collector 並加密。\n"
            f"- 供應商遠端維運走核准跳板與時限帳號。\n"
        )
        return key, instruction, output

    if re.search(r"BOOT|INSTALL|IOSXE|PLATFORM|IMAGE|UPGRADE", hay):
        instruction = f"請從弱點／韌體管理角度分析（{SITE}）：\n{raw.strip()}"
        output = (
            f"## 地端 LLM 智慧合規診斷報告\n\n"
            f"## 一、事件經過摘要\n"
            f"`{code}` 與開機、映像、平台或安裝升級有關。\n\n"
            f"## 二、不合規／風險分析\n"
            f"- 控制項：{meta['iso']} {meta['title']}\n"
            f"- 未驗證映像或延遲修補會擴大已知 CVE 暴露面。\n\n"
            f"## 三、具體修補建議\n"
            f"- 僅安裝簽章映像；維護窗口升級；保留前一版可回退。\n"
            f"- 追蹤 Cisco PSIRT 並更新弱點清冊。\n"
        )
        return key, instruction, output

    if re.search(r"LINK-.*UPDOWN|LINEPROTO-.*UPDOWN|\bUPDOWN\b", hay) or mnemonic == "UPDOWN":
        iface_m = re.search(
            r"Interface\s+([A-Za-z0-9/.-]+)",
            body,
            re.I,
        )
        iface = iface_m.group(1) if iface_m else "（未標示埠位）"
        instruction = (
            f"請分析此介面狀態變更／抖動（link flap）日誌（{SITE}）：\n{raw.strip()}"
        )
        output = (
            f"## 地端 LLM 智慧合規診斷報告\n\n"
            f"## 一、事件經過摘要\n"
            f"`{code}`：交換器埠位 `{iface}` 出現 link/line-protocol "
            f"**up ↔ down** 狀態切換。這是**實體／二層連線可用性事件**"
            f"（介面抖動 link flap），不是組態檔新增刪除，也不是控制器 MFG 改配方，"
            f"更不是重放攻擊。\n"
            f"若同一埠位短時間多次 up/down，應優先當抖動故障排查。\n\n"
            f"## 二、不合規／風險分析\n"
            f"- 控制項對應：可用 A.8.19（變更／營運事件稽核）追蹤，"
            f"本質風險是**製程網可用性與連線完整性**。\n"
            f"- 常見原因：線材／SFP／接頭鬆脫、對端設備重啟、PoE 不足、"
            f"錯誤 speed/duplex、環路或 errdisable 後恢復。\n"
            f"- 半導體廠現場可能影響設備通訊、AMHS／機台 eth 連線中斷。\n"
            f"- **禁止**解讀成「控制器新增／刪除配置檔」或無關 ICS 攻擊劇本。\n\n"
            f"## 三、具體修補建議\n"
            f"- 現場檢查 `{iface}` 線材、模組、對端狀態與 errdisable 原因。\n"
            f"- 核對介面 speed/duplex/udld；PoE 埠確認電力預算。\n"
            f"- 對抖動埠設監控告警（短時間 N 次 UPDOWN）；必要時先隔離測試。\n"
            f"- 若為維護拔線，應有變更窗口紀錄，避免與故障混淆。\n"
        )
        return key, instruction, output

    if re.search(r"ILPOWER|POWER_|ERR_DISABLE|SPANTREE", hay):
        instruction = f"請分析此介面供電／生成樹／errdisable 相關事件（{SITE}）：\n{raw.strip()}"
        output = (
            f"## 地端 LLM 智慧合規診斷報告\n\n"
            f"## 一、事件經過摘要\n"
            f"`{code}`：{body[:220]}\n"
            f"屬交換器埠位供電、生成樹或錯誤停用類事件，影響連線可用性。\n\n"
            f"## 二、不合規／風險分析\n"
            f"- 控制項：{meta['iso']} {meta['title']}\n"
            f"- PoE 不足或 errdisable 會造成現場設備掉線。\n\n"
            f"## 三、具體修補建議\n"
            f"- 查 `show interface status` / errdisable 原因；調整 PoE 預算或排除環路。\n"
            f"- 復原前確認不影響安全聯鎖與製程通訊。\n"
        )
        return key, instruction, output

    # 預設：其餘 syslog → 依實際訊息解讀，禁止套劇本
    instruction = (
        f"請分析下列 {SITE} Cisco OT 交換器 syslog，並對應 ISO 27001 控制項：\n{raw.strip()}"
    )
    output = (
        f"## 地端 LLM 智慧合規診斷報告\n\n"
        f"## 一、事件經過摘要\n"
        f"設備產生 `{code}`。內容：{body[:220]}\n"
        f"請嚴格依 facility/mnemonic 與正文解讀，勿虛構設備名稱或無關攻擊。\n\n"
        f"## 二、不合規／風險分析\n"
        f"- 建議控制項：{meta['iso']} {meta['title']}\n"
        f"- 評估對製程網可用性與基線偏離的影響。\n\n"
        f"## 三、具體修補建議\n"
        f"- 對照變更／維運窗口；必要時隔離埠位並保全日誌。\n"
        f"- 對重複異常設告警並追蹤結案。\n"
    )
    return key, instruction, output


def load_ot_log_docs(per_type: int, max_docs: int) -> list[dict]:
    files = sorted(
        p for p in OT_DIR.glob("*.txt")
        if not p.name.lower().startswith("expand_")
    )
    if not files:
        print(f"⚠️ OT 目錄無 txt：{OT_DIR}")
        return []

    buckets: dict[str, list[str]] = defaultdict(list)
    for path in files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.rstrip("\n")
                m = CISCO_RE.search(line)
                if not m:
                    continue
                facility, sev, mnemonic, body = m.group(1), m.group(2), m.group(3), m.group(4)
                type_key = f"{facility}-{mnemonic}"
                if len(buckets[type_key]) >= per_type:
                    continue
                # 去重（同 type 內 raw 太像就跳過）
                if any(line[-80:] == x[-80:] for x in buckets[type_key]):
                    continue
                buckets[type_key].append(line)

    docs = []
    idx = 0
    for type_key in sorted(buckets.keys()):
        for raw in buckets[type_key]:
            if len(docs) >= max_docs:
                break
            m = CISCO_RE.search(raw)
            if not m:
                continue
            facility, sev, mnemonic, body = m.group(1), m.group(2), m.group(3), m.group(4)
            key, instruction, output = analysis_for_event(
                facility, sev, mnemonic, body, raw
            )
            docs.append(
                make_doc(
                    f"ot_{idx:05d}",
                    instruction,
                    output,
                    "ot_log",
                    tags=[key, f"%{facility}-{mnemonic}", CONTROL_META[key]["iso"]],
                )
            )
            idx += 1
        if len(docs) >= max_docs:
            break

    print(f"✓ ot/*.txt → {len(docs)} 篇（{len(buckets)} 種訊息類型）")
    return docs


def curated_control_docs() -> list[dict]:
    """固定幾篇「易誤判」校正知識，壓過舊 KB 的重放攻擊模板。"""
    items = [
        (
            "Cisco %SEC_LOGIN-5-LOGIN_SUCCESS 是否代表重放攻擊？",
            "否。`SEC_LOGIN-*-LOGIN_SUCCESS` 是**交換器／路由器管理登入成功**事件，"
            "對應 ISO 27001 存取控制（A.7.4／A.11.2），**不是** ICS 控制指令重放。"
            "重放攻擊應看工控協議（如 Modbus 寫入重送、OPC UA 無 nonce）與 IDS 簽章，"
            f"不能僅憑 LOGIN_SUCCESS 下結論。在{SITE}應檢查：帳號是否共用、"
            "通道是 SSH 還是 Telnet、來源 IP 是否在網管白名單。",
        ),
        (
            "如何區分管理面登入日誌與 OT 控制指令異常？",
            "管理面：`SEC_LOGIN`、`SSH`、`TACACS`、`RADIUS`、`LOGOUT`、`TTY_EXPIRE` → 存取控制。\n"
            "傳輸安全：`SNMP`、`CRYPTO`、`PKI`、`IKE` → A.8.24。\n"
            "組態：`CONFIG_I`、`PARSER` → A.8.19。\n"
            "惡意：`MALWARE`、`IPS`、`PORT_SECURITY` → A.8.7。\n"
            "只有出現工控寫入／異常功能碼／未授權 setpoint 變更時，才討論重放或非法控制。",
        ),
        (
            f"{SITE} Cisco Catalyst 管理面最小加固清單",
            "1. transport input ssh（禁用 Telnet）\n"
            "2. AAA 個人帳 + 指令授權稽核\n"
            "3. 管理 ACL／專用 VRF\n"
            "4. SNMPv3、關閉弱 community\n"
            "5. exec-timeout、logging 集中、組態 archive\n"
            "6. 定期映像修補（A.8.8）",
        ),
        (
            "日誌同時有 LOGIN_SUCCESS 與 TTY_EXPIRE_TIMER、LOGOUT 代表什麼？",
            "代表完整的管理 session 生命週期：登入成功 →（閒置）逾時 → 登出。"
            "這通常說明已設定 session timeout，屬正面控制；重點改放在帳號身分、"
            "來源位址與是否使用明文 Telnet，而不是判定為攻擊重放。",
        ),
        (
            "%LINK-3-UPDOWN Interface changed state to down/up 代表什麼？",
            "`%LINK-*-UPDOWN` 表示交換器**實體埠位 link 狀態在 up 與 down 之間切換**。"
            "同一埠位短時間多次出現即為 **link flap（介面抖動）**。"
            "常見原因是線材、SFP、對端重啟、PoE 或協商異常。"
            "這不是組態檔新增刪除，也不是 MFG 控制器改配方，不應寫成重放攻擊。"
            "排查：檢查該 Interface、線材／模組、errdisable、speed/duplex，並對抖動設告警。",
        ),
        (
            "C9300 GigabitEthernet 連續 UPDOWN 如何做合規診斷？",
            "## 一、事件經過摘要\n"
            "Catalyst 埠位（如 Gi1/0/3）連續 down→up，屬介面抖動／可用性事件。\n\n"
            "## 二、不合規／風險分析\n"
            "- 風險在製程通訊中斷與監控完整性；控制項可掛 A.8.19 事件稽核。\n"
            "- 禁止解讀為控制器配置檔變更或 ICS 指令攻擊。\n\n"
            "## 三、具體修補建議\n"
            "- 現場查線與對端；確認 PoE／協商；監控短時間 UPDOWN 次數並留存證據。",
        ),
    ]
    docs = []
    for i, (q, a) in enumerate(items):
        docs.append(
            make_doc(
                f"curated_{i:03d}",
                q,
                a,
                "curated",
                tags=["access_control", "anti_hallucination", "A.7.4"],
            )
        )
    # 每篇複製一次換問法，提高檢索命中
    extras = []
    for i, d in enumerate(docs):
        extras.append(
            make_doc(
                f"curated_x_{i:03d}",
                f"【{SITE}】{d['instruction']}",
                d["output"],
                "curated",
                tags=d["tags"],
            )
        )
    print(f"✓ curated 校正知識 → {len(docs) + len(extras)} 篇")
    return docs + extras


def load_filtered_old_kb(limit: int) -> list[dict]:
    if not KB_PATH.is_file() or limit <= 0:
        return []
    try:
        old = json.loads(KB_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠️ 讀舊 KB 失敗：{e}")
        return []

    keep = []
    for i, row in enumerate(old):
        q = (row.get("instruction") or row.get("title") or "").strip()
        a = (row.get("output") or row.get("content") or row.get("text") or "").strip()
        if not q or not a:
            continue
        blob = q + "\n" + a
        # 剔除異場域
        if FOREIGN_RE.search(blob):
            continue
        # 只要與本平台相關或通用 OT／ISO
        if not re.search(
            r"半導體|ISO\s*27001|Cisco|SEC_LOGIN|RADIUS|TACACS|SNMP|Modbus|"
            r"OPC\s*UA|PLC|IEC\s*62443|syslog|存取控制|組態|惡意|弱點|供應商",
            blob,
            re.I,
        ):
            continue
        # 明確剔除「只用 LOGIN 卻講重放」這類有害模板：保留一般重放攻擊教材，
        # 但後面 curated／ot 文件會用更高相關度蓋過；此處若標題同時含 LOGIN 與重放則丟棄
        if re.search(r"LOGIN_SUCCESS|SEC_LOGIN", blob, re.I) and re.search(
            r"重放攻擊|Replay", blob, re.I
        ):
            continue
        keep.append(
            make_doc(
                f"old_{i:05d}",
                q,
                a,
                "old_kb_filtered",
                tags=["legacy_filtered"],
            )
        )
        if len(keep) >= limit:
            break
    print(f"✓ 舊 KB 過濾保留 → {len(keep)} 篇（上限 {limit}）")
    return keep


def dedupe(docs: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for d in docs:
        sig = re.sub(r"\s+", " ", (d.get("instruction") or "")[:200]).lower()
        if sig in seen:
            continue
        seen.add(sig)
        out.append(d)
    return out


def backup_kb():
    if not KB_PATH.is_file():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = ROOT / f"knowledge_base.backup_{ts}.json"
    shutil.copy2(KB_PATH, bak)
    print(f"📦 已備份舊 KB → {bak.name}")
    return bak


def write_outputs(docs: list[dict]):
    KB_PATH.write_text(
        json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pairs = [{"query": d["instruction"], "positive": d["output"]} for d in docs]
    random.shuffle(pairs)
    with open(PAIRS_PATH, "w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"✅ 寫入 {len(docs)} docs → {KB_PATH}")
    print(f"✅ 寫入 {len(pairs)} pairs → {PAIRS_PATH}")


def rebuild_index():
    from build_index import main as build_main

    print("🔁 重建向量索引...")
    build_main()


def parse_args():
    p = argparse.ArgumentParser(description="從目前資料建立 RAG 資料集（預設=log 分析）")
    p.add_argument("--per-type", type=int, default=10, help="每種 syslog 抽樣幾筆")
    p.add_argument("--max-ot-docs", type=int, default=1200, help="OT 日誌最多轉幾篇")
    p.add_argument("--old-kb-limit", type=int, default=200, help="舊 KB 過濾後最多保留")
    p.add_argument(
        "--include-train",
        action="store_true",
        help="合併 train.json（偏理論；log 診斷建議不要加）",
    )
    p.add_argument(
        "--include-old-kb",
        action="store_true",
        help="合併過濾後舊 KB（易混入非 log 分析）",
    )
    p.add_argument("--no-old-kb", action="store_true", help="相容舊參數（等同預設）")
    p.add_argument("--rebuild-index", action="store_true", help="完成後重建 index")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    print("=" * 60)
    print(f"建立 RAG 資料集｜場域={SITE}｜模式=log 分析優先")
    print("=" * 60)

    backup_kb()

    # 預設只要「log → 診斷答案」，避免 Modbus/重放攻擊等理論蓋過 syslog 分析
    docs: list[dict] = []
    docs.extend(curated_control_docs())
    docs.extend(load_ot_log_docs(args.per_type, args.max_ot_docs))
    if args.include_train:
        docs.extend(load_train_json())
    if args.include_old_kb and not args.no_old_kb:
        docs.extend(load_filtered_old_kb(args.old_kb_limit))

    docs = dedupe(docs)
    for i, d in enumerate(docs):
        d["id"] = f"doc_{i:05d}"
        # 明確標記用途，供檢索加權
        d["doc_type"] = "log_analysis"

    by_src = defaultdict(int)
    for d in docs:
        by_src[d.get("source", "?")] += 1

    write_outputs(docs)
    print("來源統計：", dict(by_src))
    print("（此 KB 專供 syslog／合規 log 分析，不含一般理論問答）")

    if args.rebuild_index:
        rebuild_index()
    else:
        print("提示：執行 python build_index.py 以更新向量；或加 --rebuild-index")


if __name__ == "__main__":
    main()
