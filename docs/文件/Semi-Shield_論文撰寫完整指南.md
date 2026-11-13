# Semi-Shield ISMS 論文撰寫完整指南

> **用途：** 撰寫大專／研究計畫／期刊稿時的「內容地圖」— 說明專案做什麼、怎麼做、量到什麼、該怎麼寫進各章。  
> **搭配文件：** `Semi-Shield_ISMS_大專研究計畫論文.md`（緒論已寫較細）、`Semi-Shield_專案實驗與成果總說明.md`（數字快照）、`Semi-Shield_Nature_Format_Manuscript.md`（英文稿結構）。  
> **圖表素材：** `data/eval_sac/paper/paper_tables_figures.html`、`data/eval_sac/figures/table*.png`、`figure3–5*.png`。

---

## 零、30 秒電梯簡報（可放摘要第一段）

Semi-Shield ISMS 是面向半導體 OT 場域的 **ISO/IEC 27001 Annex A 合規輔助平台**，設計原則為 **「No Evidence, No Compliance Claim」**。系統將 **Cisco 風格 syslog**、**政策／GRC 人工匯入**、**Evidence Registry（SHA-256 hash chain）**、**地端 LLM（Ollama）** 與 **Human Review Queue** 串成可演示的證據鏈；並提出 **CORG（Control-Obligation Release Gate）**，以 **義務閉包（obligation read-set）** 約束合規判斷與證據釋出。在 522 筆 OT 情境評估中，CORG 將 **誤肯定率（FAR）由 40.8% 降至 0%**，並使 **未授權證據釋出由 100% 降至 0%**（相對 citation-only 綁定）。

---

## 一、論文建議章節結構（可直接當目錄）

| 章 | 建議標題 | 本章要回答什麼 | 主要素材來源 |
|----|----------|----------------|--------------|
| 1 | 緒論 | 為何 OT-ISMS 需要證據鏈？研究問題？貢獻？ | `Semi-Shield_ISMS_大專研究計畫論文.md` §1 |
| 2 | 文獻探討與相關工作 | GRC、OT 安全、LLM 稽核、RAG 幻覺、release policy | 計畫書 §2 + 自行補 NIST 800-82、ISO 27001 |
| 3 | 系統需求與設計 | 雙軌 93 項、管線、資料模型 | `ARCHITECTURE.md`、本指南 §三 |
| 4 | 證據鏈與 CORG 方法 | Registry、Review、CORG 演算法 | 本指南 §四、`obligations.py` |
| 5 | 系統實作 | 模組、API、前端、部署 | `PROJECT_LAYOUT.md`、§3.3 模組表 |
| 6 | 實驗設計與評估 | B0–B5、SVI、release 電池 | 本指南 §五、`code/eval/` |
| 7 | 結果與討論 | Table 1–3、Figure 3–5、限制 | `results.json`、論文圖目錄 |
| 8 | 結論與未來工作 | 貢獻重申、Phase 2 | 計畫書 Phase 2、§七限制 |

**英文稿／Nature 結構：** 可將第 6–7 章合併為 *Methods*（文末）+ 發現導向正文，參考 `Semi-Shield_Nature_Format_Manuscript.md`。

---

## 二、研究問題與論文主張（對照用）

撰寫時建議在緒論末尾用表格對照「問題 → 系統機制 → 評估方式 → 目前答案」：

| 研究問題 | 系統回應 | 如何驗證 | 目前結論（本 repo） |
|----------|----------|----------|---------------------|
| **RQ1** 如何避免 AI／檢索「誤判合規」？ | CORG `corg_assess` + scope/signal/coverage | Table 4 FAR、Figure 3 | B5 FAR=0；B0 FAR≈57% |
| **RQ2** 證據如何可追溯、可稽核？ | `evidence_registry.jsonl` hash chain | 架構說明 + registry 統計 | 965 筆；payload_hash + prev_hash |
| **RQ3** Citation 是否足夠安全釋出？ | obligation read-set digest 綁定 | Table 2、Figure 5 | AGT-cited 100% vs AGT+CORG 0% |
| **RQ4** 93 項 Annex A 如何落地？ | auto_syslog（6）+ auto_import（87） | evidence_dashboard、UI | 雙軌證據鏈 SVG |
| **RQ5** 地端 LLM 如何被約束？ | RAG + 戰情上下文 + Guardrail + Queue | 管線描述、注入實驗 | 注入成功率 B5 66.7% vs B2 83.3% |

**論文主張（建議寫進 1.5 節「研究貢獻」）：**

1. **方法貢獻：** 提出 CORG，將合規判斷與證據釋出綁定到 **可執行的義務閉包**，而非僅 LLM 引用或靜態 citation list。  
2. **工程貢獻：** 可運行之 OT-ISMS 原型（戰情室、拓樸、Registry、Review、Agent）。  
3. **實證貢獻：** 可重現評估管線（`scripts/run_sac_corg_evaluation.py`）與論文級圖表（`export_sac_paper_figures_tables.py`）。

---

## 三、系統內容詳解（第三章～第五章素材）

### 3.1 問題域與設計原則

- **標準：** ISO/IEC 27001:2022 **Annex A**（本系統 UI 呈現 **93 項**控制項視圖：矩陣 71 項 + A.6 人員 + A.7 實體補齊）。  
- **場域：** 半導體 Fab OT；**Purdue L0–L3** 拓樸（`ot_topology_service`、`config/ot_topology_sites.json`）。  
- **原則：** **No Evidence, No Compliance Claim** — LLM 與報表不得脫離 `evidence_id` 與 bundle 原文。  
- **部署：** 地端 Flask `:2000` + Ollama `:11434`；syslog 留在 `ot/`，符合 **log 不出廠** 需求。

### 3.2 整體架構（建議放一張圖）

```
[ ot/*.txt syslog ] ──► Collector（掃描／分類）
         │                      │
         ▼                      ▼
[ manual_imports ]      Control Bundle + KPI
         │                      │
         └──────────┬───────────┘
                    ▼
         Evidence Registry (hash chain)
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
 AI Auditor     Review Queue    Compliance UI
 (Ollama+RAG)   (Human/AI)      (93 項證據鏈)
                    │
                    ▼
              CORG（評估／釋出語意）
```

**論文可引用圖：** `data/eval_sac/figures/figure1_architecture.png`（評估架構）、戰情室截圖（`系統截圖/`）。

### 3.3 雙軌自動化（必寫，區隔與 GRC 差異）

| 層級 | 控制項數 | 資料來源 | 模組 |
|------|----------|----------|------|
| **auto_syslog** | 6 | `ot/` 設備 log | `scan_ot_directory` → 六 `control_key` |
| **auto_import** | 87 | 政策／HR／實體等 | `manual_import_service`、`simulate_demo_batch` |

六項 OT 監控對照（**論文表格必列**）：

| control_key | Annex A | 監控語意 |
|-------------|---------|----------|
| access_control | A.5.15 | 管理面登入／AAA |
| supplier_security | A.5.19 | CDP/LLDP 鄰居 |
| malware_defense | A.8.7 | IPS／惡意事件 |
| patch_management | A.8.8 | 版本／CVE 提示 |
| recipe_audit | A.8.19 | 組態變更 CONFIG_I |
| sec_gem_log | A.8.24 | SNMP／PKI／crypto |

其餘 Annex A 在戰情室以 **AI 匯入管線敘事** + `manual_imports.jsonl` 示範；論文需說明 **Phase 1 不宣稱 93 項皆 syslog 自動化**。

### 3.4 Agent 四階段管線（方法章節）

`agent_orchestrator` / `POST /api/compliance/pipeline`：

1. **Collector** — 掃 OT、產 bundle  
2. **Auditor** — RAG + LLM 診斷（`/api/audit/analyze`）  
3. **Reviewer** — Guardrail、AI Reviewer、Review Queue  
4. **Reporter** — PDF/TXT（`OT.html` 匯出）

**AI Chat**（`/api/agent/chat`）另路徑：護欄 → 強制 RAG → 注入與 OT.html **相同**之戰情上下文 → LLM → 幻覺護欄。論文應強調：**對話與戰情室共用事實來源**，避免「聊天機器人另一套真相」。

### 3.5 前端與演示（實作章節／附錄）

| 頁面 | 論文可描述之功能 |
|------|------------------|
| `OT.html` | KPI、事件串流、證據鏈 SVG、AI 診斷、稽核 PDF |
| `ot_evidence.js` | Semi-Shield vs 傳統 9 步流程圖 |
| `compliance.html` | 矩陣覆蓋率、GRC 定位 |
| `ot_topology.html` | 多 site、設備與 log 空間上下文 |
| `platform.html` | 四窗格整合演示 |

---

## 四、證據鏈與 CORG（第四章核心）

### 4.1 Evidence Registry（執行期「上鏈」）

- **檔案：** `data/evidence_registry.jsonl`（append-only，一行一筆）。  
- **模組：** `code/services/evidence_service.py`。  
- **觸發：** 每次監控資料刷新，`_attach_evidence_and_reviews()` 對各 **control bundle** 呼叫 `register_control_bundle_evidence()`。

**一筆 evidence 包含：**

- `evidence_id`：`EV-{site}-{control_key}-{UTC}-{hash8}`  
- `payload`：標題、log 摘錄（約 500 字）、Annex A、event_count、metric_summary  
- `payload_hash`：SHA-256(canonical JSON)  
- `prev_hash`：前一筆 registry 的 payload_hash（genesis 為 64 個 0）  
- `review_status`：pending → approved/rejected（`update_evidence_review_status`）

**論文要寫清楚的限制（誠實加分）：**

- Hash chain 為 **append 日誌鏈**，非區塊鏈共識；審核狀態更新會 **重寫整檔**。  
- 每次刷新可能 **新增** evidence 列，而非原地更新同一 KPI 版本（可寫成 future work：版本化 evidence）。

### 4.2 Human Review Queue

- **檔案：** `data/review_queue.jsonl`  
- metrics 為 **review** 或 **fail** 時自動 `enqueue_review`  
- **AI Reviewer**（`ai_reviewer_service.py`）：檢查是否有 evidence_id／payload_hash；fail 狀態可核准為「帶證據之待修復」，非假裝合規

### 4.3 93 項「證據鏈視圖」（UI／教學層）

- **模組：** `evidence_chain_service.py` + `control_evidence_profiles.py`  
- **API：** `GET /api/monitor/evidence_dashboard`  
- 每控制項輸出：`semi_shield_chain`（9 步）、`traditional_chain`（對照）、`sources[]`、`log` 原文  

**OT 自動化鏈（論文可逐字引用步驟）：**  
Syslog 來源 → Collector 解析 → 分類桶 → Control Bundle → **Evidence Registry** → AI Auditor → Guardrail + AI Reviewer → **Human Review Queue** → 合規宣稱  

### 4.4 CORG 演算法（方法章必寫公式化描述）

**實作：** `code/eval/obligations.py`（評估管線與 Semi-Shield 語意一致；生產路徑可對照擴充）。

**步驟 1 — 閉包選取器** `execute_closure_selectors`：

- **scope:eligible_records**：trusted 且 site/asset 符合稽核範圍  
- **signal:positive**：control_key 匹配且 log 含 positive_signals（如 SEC_LOGIN）  
- **signal:violation**：severity 或 violation_signals（如 LOGIN_FAILED）  
- **coverage:collector**：collector 是否覆蓋該 scope  

**步驟 2 — 判斷** `corg_assess` → SVI：

- 有 violation → **VIOLATED**  
- 有 positive 且 coverage OK 且 cited span 足夠 → **SUPPORTED**  
- 否則 → **INSUFFICIENT**  

**步驟 3 — 釋出綁定**（`release_eval.py`）：

- **AGT-cited**：報告綁定 approved citation bundle digest  
- **AGT+CORG**：報告綁定 **obligation_read_set_digest**（閉包摘要）  
- 反證寫入後 digest 改變 → 未授權釋出  

**論文可用一句話：** CORG 將「可以對外說合規」與「可以對外釋出哪些紀錄」都約束在同一 **義務讀取集** 上。

---

## 五、實驗設計（第六章素材）

### 5.1 評估目的

- 比較 **B0～B5** 在 OT 證據情境下之 **FAR、Resolution、Macro-F1**  
- 比較 **release 綁定** 在 counterevidence 變異下之 **未授權�釋出率**  
- 補充：κ（標註一致性）、注入、跨域、延遲（bundle）

### 5.2 資料集

- **來源：** 真實 `ot/` log pool 抽樣 + 合成變體（`scenarios.py`）  
- **規模（核心 run）：** 120 groups、**522 packets**  
- **Gold SVI：** SUPPORTED 240 / VIOLATED 42 / INSUFFICIENT 240  
- **六 family：** 對應六 Annex A obligation templates  

**變體（寫進方法，展現嚴謹）：**

| variant | 目的 |
|---------|------|
| clean | 正常支持 |
| clean-violated | 違規訊號 |
| evidence_removed | 證據不足 |
| scope_mismatch | 範圍錯誤 Fab |
| injection | 不可信紀錄 prompt 注入 |

**限制（討論章必寫）：** Gold 為 **rubric／規則標註**，非第三方專家盲審；κ 0.885 為 **模擬雙標**（B=gold+7% 噪聲）。

### 5.3 基線定義（Table 1 文字版）

| 代號 | 論文描述 |
|------|----------|
| B0 | 樂觀稽核草稿（無 grounded，有紀錄即 SUPPORTED） |
| B0-full | 義務感知保守基線（無 CORG 釋出語意） |
| B2 | + 關鍵字檢索 |
| B3 | + 同 site 圖擴展（stub） |
| B4 | + span／control 過濾（Semi-Shield 無 CORG） |
| **B5** | **+ CORG 完整 assess** |

### 5.4 指標定義（給審查委員）

- **FAR** = #{gold≠SUPPORTED 但 pred=SUPPORTED} / #{gold≠SUPPORTED}  
- **Resolution** = #{gold 可回答且 pred 正確 SVI} / #{gold 可回答}  
- **未授權釋出率** = release 試次中 binding 失效仍釋出之比例  

### 5.5 統計方法

- **B5 vs B4 FAR：** 以 **group 為單位** 配對 bootstrap（10,000 draws），95% CI  
- 圖：**Figure 4** histogram；若 CI 不含 0 且單尾 p≈0 → 報 **P < 0.001**

### 5.6 重現指令（附錄）

```bat
py -3.12 scripts\run_sac_corg_evaluation.py --groups 20
py -3.12 scripts\run_sac_full_data_generation.py --groups 20
py -3.12 scripts\export_sac_paper_figures_tables.py
```

---

## 六、結果撰寫指引（第七章：可直接改寫進正文）

### 6.1 判斷效能（對應 Table 1、Figure 3）

**建議敘事順序：**

1. 先報 **B0 FAR ≈ 57.5%** — 確立「無治理的 AI／草稿」高風險。  
2. **B2–B4 仍 ≈ 41% FAR** — 檢索 alone 不足。  
3. **B5 FAR = 0%**，Resolution **0.504**，Macro-F1 **0.653** — 與 B0-full 相同，代表 **安全收斂**；相對 B4 **降幅 40.8 個百分點**（Figure 4）。  
4. Figure 3：**B0–B4 落在 FAR>40% 危險區**；B5 貼在 FAR=0 軸上。

**不要過度宣稱：** B3 相對 B2 resolution **下降** — 需討論 graph stub 或 OT 雜訊，不宜寫「圖檢索全面優於平面檢索」。

### 6.2 釋出安全（Table 2、Figure 5）

- **AGT-cited：100%** 未授權釋出 → 傳統「只 cite 附件」在 OT 反證情境失敗。  
- **AGT+CORG：0%** → read-set 綁定有效。  
- **irrelevant still release** 各策略皆高 — 可列 **限制與後續工作**。

### 6.3 強健性與效能（Table 3）

- Prompt 注入：B5 **66.7%** vs B2 **83.3%**（仍非零，需 hardening）  
- 日誌置換：約 **83.3%** 未維持 SUPPORTED（回退 INSUFFICIENT）  
- 跨域 synthetic：B5 **FAR 0%** 但 resolution **0%**（保守拒判）  
- 延遲：B5 p50 **0.002 ms/packet**（規則引擎；**不含** Ollama E2E）

### 6.4 系統運行數據（可放「個案／部署」小節）

來自 `data/資料總覽.md`：

- evidence_registry **965** 筆；review_queue **895** 筆；manual_imports **99** 筆（示範）  
- 彙總 syslog 事件約 **1,044 萬**（aggregate metric）

---

## 七、限制、威脅有效性與未來工作（討論章 checklist）

撰寫時逐條回應，展現研究成熟度：

1. **Gold 標籤** — 規則合成；需專家 adjudication。  
2. **κ** — 模擬雙標；非 inter-rater 研究。  
3. **Ollama B2 writer** — bundle 中可能為 null；端到端 LLM 實驗待補。  
4. **Registry** — append 多版本、非 Merkle audit log。  
5. **93 項** — 87 項靠 import 示範，非全自動 SIEM。  
6. **跨域** — synthetic water/power，非真實 Fab 跨 site。  
7. **認證** — 不宣稱通過 ISO 27001 正式驗證。

**Phase 2（計畫書可寫）：** Wazuh 規則擴充、OpenCTI、真實 GRC API、evidence 版本化、正式 expert study。

---

## 八、圖表與附錄清單（投稿／口試）

| 編號 | 內容 | 檔案 |
|------|------|------|
| 圖 1 | 系統／評估架構 | `figure1_architecture.png` |
| 圖 2 | CORG 釋出流程 | `figure2_corg_release.png` |
| 圖 3 | FAR–Resolution | `figure3_risk_resolution_scatter.png` |
| 圖 4 | Bootstrap CI | `figure4_bootstrap_far_ci.png` |
| 圖 5 | Release 柱狀圖 | `figure5_release_security.png` |
| 表 1 | 基線對照 | `table1_baseline_comparison.png` |
| 表 2 | Release | `table2_release_security.png` |
| 表 3 | 強健性／延遲 | `table3_robustness_latency.png` |
| 附錄 A | API 列表 | `ARCHITECTURE.md` §2.2 |
| 附錄 B | evidence_schema | `compliance/evidence_schema.json` |
| 附錄 C | 評估 manifest | `scenario_manifest.json` |

**整合 HTML：** `data/eval_sac/paper/paper_tables_figures.html`

---

## 九、摘要（中文）範本（約 300 字，可調）

半導體營運技術（OT）場域在 ISO/IEC 27001 稽核中須對 Annex A 控制項提出可驗證證據，惟傳統以人工匯出 syslog、Excel 對照與 PDF 快照為主之流程，存在版本分散、結論與原始 log 脫鉤及生成式 AI 易產生「無證據合規宣稱」等風險。本研究設計並實作 Semi-Shield ISMS 平台，以「No Evidence, No Compliance Claim」為原則，整合 OT syslog 自動監控、93 項 Annex A 雙軌視圖、SHA-256 Evidence Registry、Human Review Queue 與地端 LLM 診斷，並提出 CORG（Control-Obligation Release Gate）機制，以義務閉包約束合規判斷與證據釋出。實驗於 522 筆由真實 OT 語料衍生之情境封包上，比較 B0 至 B5 基線：未受約束之樂觀草稿誤肯定率達 57.5%，而 CORG 將誤肯定率降至 0%，Bootstrap 95% 信賴區間顯示相較無 CORG 之 Semi-Shield 降幅約 40.8 個百分點；在反證寫入試驗中， citation-only 釋出策略未授權洩漏率為 100%，CORG 讀取集綁定則為 0%。結果顯示 OT 合規自動化應同時以誤肯定率與釋出安全為評估核心。限制包括 gold 標籤為規則合成及部份控制項仍依示範匯入，未来將進行專家標註與端到端 LLM 評估。

---

## 十、程式與文件索引（寫作時快速查）

| 主題 | 路徑 |
|------|------|
| 啟動 | `run_ollama.bat`、`app.py` |
| 證據註冊 | `code/services/evidence_service.py` |
| 證據鏈 UI 資料 | `code/services/evidence_chain_service.py` |
| CORG 評估 | `code/eval/obligations.py`、`baselines.py`、`release_eval.py` |
| 實驗結果 | `data/eval_sac/results.json` |
| 完整 bundle | `data/eval_sac/full_data_bundle.json` |
| Word 論文產生 | `scripts/build_isms_plan_paper_docx.py` |
| 計畫書長稿 | `docs/文件/Semi-Shield_ISMS_大專研究計畫論文.md` |

---

*本指南隨 codebase 更新；重跑實驗後請同步修訂 §六數字與圖表。*
