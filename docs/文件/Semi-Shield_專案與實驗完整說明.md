# Semi-Shield ISMS — 專案詳細資訊與實驗完整說明

> 產生日期：2026-10-18  
> 資料來源：`data/eval_sac/results.json`、`data/eval_sac/full_data_bundle.json`、`data/資料總覽.md`、現行 codebase  
> 用途：計畫書／論文／決賽簡報之「專案＋實驗」單一參考文件

---

## 一、專案定位

**Semi-Shield ISMS** 是面向 **OT（營運技術）工控環境** 的 **ISO/IEC 27001 Annex A** 合規與稽核輔助平台。核心設計原則：

**No Evidence, No Compliance Claim（無證據、不宣稱合規）**

系統將 Cisco 等設備 **syslog**、手動匯入的 **ISMS 政策／紀錄**、以及 **RAG／地端 LLM** 診斷，收斂為可稽核的 **Evidence（證據）** 與 **Review Queue（覆核佇列）**。在 SAC ’27 評估協議子集上，另實作 **CORG（Control-Obligation Release Gate）**：以 **義務閉包（obligation read-set）** 綁定「可釋出之證據集合」，降低 **誤肯定（False Affirmative）** 與 **未授權證據釋出**。

| 項目 | 說明 |
|------|------|
| 預設服務埠 | **2000**（Flask） |
| LLM 後端 | **Ollama**（`:11434`，可切換本地 Transformers） |
| 主要語言 | 後端 Python 3.12、前端 HTML/JS |
| 專案根目錄 | `C:\Users\Max\Desktop\ot_test`（執行時以 `code/paths.py` 解析路徑） |

---

## 二、系統架構（分層）

```
瀏覽器 (http://127.0.0.1:2000)
  platform.html │ OT.html │ agent_chat.html │ compliance.html │ ot_topology.html
        ↓ REST / 靜態
code/server/app.py (Flask)
  監控掃描 · Agent 對話 · 合規管線 · 證據 API · 拓樸 API
        ↓
code/services/*
  ollama · rag · guardrail · hallucination · evidence · compliance · review · evidence_chain · ot_topology
        ↓
data/*.jsonl │ ot/*.txt │ compliance/*.json │ config/*.json │ train_ai/*
```

### 2.1 前端頁面

| 頁面 | 路由 | 功能 |
|------|------|------|
| platform.html | `/`、`/platform` | 整合入口：戰情＋對話 |
| OT.html | `/monitor` | 六控制項 KPI、事件串流、地端診斷、匯出 |
| agent_chat.html | `/chat` | AI 對話（可串流 NDJSON）、RAG＋戰情上下文 |
| compliance.html | `/compliance` | 合規矩陣、Evidence、Review Queue |
| ot_topology.html | （web 靜態） | Purdue 拓樸、多 site、syslog 視覺化 |
| ot_evidence.js | — | 證據鏈 UI 邏輯 |

### 2.2 後端服務（精要）

| 模組 | 路徑 | 職責 |
|------|------|------|
| Ollama | `code/services/ollama_service.py` | 地端 LLM chat、模型切換 |
| RAG | `code/services/rag_service.py` | 知識庫檢索 |
| 護欄 | `guardrail_service.py`、`hallucination_guard_service.py` | 輸入／輸出安全與 grounded 檢查 |
| 證據 | `evidence_service.py` | 註冊、hash chain |
| 證據鏈視圖 | `evidence_chain_service.py` | Annex A 控制項證據儀表板 |
| 覆核 | `review_queue.py`、`ai_reviewer_service.py` | 人工／AI 覆核佇列 |
| 手動匯入 | `manual_import_service.py` | ISMS 條文 JSONL 匯入 |
| OT 拓樸 | `ot_topology_service.py` | `config/ot_topology_sites.json` 多場域 |
| Agent | `agent_orchestrator.py` | Collector→Auditor→Reviewer→Reporter 管線規格 |

### 2.3 執行期資料（`data/`）

| 檔案 | 目前規模（2026-10-18 掃描） | 意義 |
|------|------------------------------|------|
| `evidence_registry.jsonl` | **965** 筆；彙總約 **1,044 萬** syslog 事件 | 每行一筆證據（Annex A、聚合 excerpt、payload_hash、prev_hash 鏈） |
| `review_queue.jsonl` | **895** 筆 | 控制項狀態待覆核／已核准／拒絕 |
| `manual_imports.jsonl` | **99** 筆（示範 simulated） | 手動匯入 Annex A 政策片段 |
| `eval_sac/` | 見下文 | SAC/CORG 可重現評估產物 |
| `data_overview.html` | — | 資料儀表板（`scripts/build_data_overview.py` 產生） |

證據覆核狀態概況：`approved` 714、`pending` 225、`rejected` 26。

### 2.4 OT 監控六控制項（與實驗 family 對齊）

| control_key | Annex A | 監控主題 |
|-------------|---------|----------|
| access_control | A.5.15 | 管理面登入／AAA |
| supplier_security | A.5.19 | 供應商／鄰居發現 |
| malware_defense | A.8.7 | IPS／惡意行為 |
| patch_management | A.8.8 | 軟體／修補 |
| recipe_audit | A.8.19 | 組態變更 |
| sec_gem_log | A.8.24 | SNMP／crypto／憑證 |

日誌來源目錄：`ot/*.txt`（Cisco syslog 樣本）。掃描結果可快取（mtime/size），供 `/api/monitor/data` 與 AI 對話注入同一戰情上下文。

### 2.5 AI 對話管線（產品面，與 B2 實驗相關）

1. 輸入護欄 → 2. **RAG**（非寒暄一律檢索）→ 3. **build_monitor_warroom_context**（與 OT.html 同源）→ 4. **Ollama/LLM** → 5. 幻覺護欄 → 6. 回覆（可圖表、可串流）。

設計要點：有 OT 資料時 **日誌優先**；不做「跳過 LLM 的假 grounded 捷徑」。

---

## 三、CORG 與評估中的「義務模型」

評估程式碼：`code/eval/obligations.py`、`code/eval/release_eval.py`。

- **ObligationTemplate**：每個 Annex 控制項定義 **positive_signals**、**violation_signals**、是否要求 **coverage**。
- **execute_closure_selectors**：在給定 site／asset／coverage 下，計算  
  `scope:eligible_records`、`signal:positive`、`signal:violation`、`coverage:collector`。
- **corg_assess（B5）**：僅在義務閉包與 cited span 條件滿足時才輸出 **SUPPORTED**；否則 **INSUFFICIENT** 或 **VIOLATED**。
- **Release 模擬（Table 5）**：
  - **AGT-cited**：報告綁定「核准時的 cite 集合」→ 語料變異後易 **未授權釋出**。
  - **AGT-global**：綁定整體 corpus digest。
  - **AGT+CORG / AGT+same_R**：綁定 **obligation read-set digest** → 變異後仍拒絕釋出（若 digest 不符）。

---

## 四、實驗目的與研究問題（SAC 可執行子集）

對應腳本：

- 核心評估：`scripts/run_sac_corg_evaluation.py`
- 全 bundle：`scripts/run_sac_full_data_generation.py`
- 圖表：`scripts/plot_sac_eval.py`、`scripts/export_sac_figures.py`

| 編號 | 問題 | 指標 | 預期方向 |
|------|------|------|----------|
| RQ1 | CORG 是否降低 **FAR**？ | Table 4 FAR | B5 &lt; B4 |
| RQ2 | 圖檢索是否提升 **resolution**？ | B3 vs B2 resolution | 協議假設為上升（本 run 見結果章節） |
| RQ3 | CORG 是否阻擋 **未授權 release**？ | Table 5 unauthorized_release_rate | AGT+CORG &lt; AGT-cited |
| 輔助 | 標註一致性 | Cohen's κ | 專家標註目標 ≥ 0.8（目前為 **模擬雙標**） |
| 輔助 | 注入／跨域／公開任務 | bundle 各節 | pilot，與 OT gold **分開報告** |

---

## 五、實驗內容（資料集怎麼來）

實作：`code/eval/scenarios.py`、`code/eval/log_bucket.py`。

### 5.1 情境生成

- **6 個 monitoring family**（對應上表六 control_key / Annex A）。
- 每 family **groups_per_family** 組（核心 run 使用 **`--groups 20` → 120 groups**）。
- 每組含多種 **variant**（封包 **ScenarioPacket**）：

| variant | 說明 | 典型 gold |
|---------|------|-----------|
| clean | 真實 ot/ 抽樣行 + coverage OK | SUPPORTED |
| clean-violated | 每 3 組之一，加合成 violation 行 | VIOLATED |
| evidence_removed | 無紀錄 | INSUFFICIENT |
| scope_mismatch | 紀錄 site 與 scope 不符 | INSUFFICIENT |
| injection | 不可信紀錄含 prompt 注入句 | SUPPORTED（gold 仍依 rubric） |

- **Gold SVI**：SUPPORTED / VIOLATED / INSUFFICIENT，由 **rubric + 合成情境** 決定（非第三方專家逐筆標註）。

### 5.2 核心 held-out run（`results.json`）

| 項目 | 值 |
|------|-----|
| actual_groups | **120** |
| actual_packets | **522** |
| Gold 分布 | SUPPORTED **240** / VIOLATED **42** / INSUFFICIENT **240** |
| 執行時間 | **2.22 s**（規則管線，不含 Ollama 全文生成） |
| dev / hold-out 設計 | manifest 目標 dev 30、held-out 120；本檔為 **held-out 子集評估** |

### 5.3 全 bundle 額外實驗（`full_data_bundle.json`）

| 區塊 | 內容 |
|------|------|
| table2_annotation | dev 30 groups + held-out 120；**κ ≈ 0.885**（A vs B，B 含 7% 標籤噪聲模擬） |
| public_probes | FEVEROUS / ContractNLI / ACE / RAGTruth **本地 pilot** |
| injection_and_approval | AgentDojo 風格注入、Loopjacking 風格替換 |
| cross_domain_water_power | 合成水務／電力各 30 包（**非** Fab-A/B 真實跨廠） |
| latency_profile | B0–B5 每包 p50/p95（毫秒級） |
| ollama_b2_writer | 本 bundle 為 **null**（未跑或未連 Ollama） |

---

## 六、基線方法（Table 4）

實作：`code/eval/baselines.py`。

| 代號 | 名稱 | 行為摘要 |
|------|------|----------|
| **B0** | 樂觀稽核草稿 | 有 control 相關紀錄即倾向 SUPPORTED → **高 FAR** |
| **B0-full** | 義務閉包保守判斷 | 無充分 positive + coverage 則 INSUFFICIENT；有 violation 則 VIOLATED |
| **B2** | + BM25 式關鍵字檢索 | 命中則 SUPPORTED，severity 低則 VIOLATED |
| **B3** | + 圖擴展（同 site 鄰居 stub） | 在 B2 上擴 2 筆鄰居 |
| **B4** | + grounding 長度過濾 | scoped 命中才 SUPPORTED，**無 CORG �格閉包** |
| **B5** | **Semi-Shield + CORG** | `corg_assess(..., require_cited_span=True)` |

### 6.1 指標定義

- **FAR（False Affirmative Rate）**：在 gold **非 SUPPORTED** 的樣本上，被判成 **SUPPORTED** 的比例。**越低越安全**。
- **Resolution**：在 **可回答（answerable）** 子集上判對的比例。**越高越完整**。
- **Macro-F1**：三類 SVI 宏平均 F1。
- **Unauthorized release rate**：核准綁定與當前 obligation／語料不一致仍釋出之比例。

---

## 七、實驗結果（核心：Table 2 / 4 / 5 / 6）

> 數值來源：`data/eval_sac/results.json`（2026-10-17 前後產生之 run）

### 7.1 Table 2 — 資料組成

| 項目 | 值 |
|------|-----|
| 封包數 n | 522 |
| 組數 | 120 |
| SUPPORTED / VIOLATED / INSUFFICIENT | 240 / 42 / 240 |

### 7.2 Table 4 — OT 判斷（FAR ↓  desirable，resolution ↑ desirable）

| 方法 | FAR | Resolution | Macro-F1 | 誤肯定數 |
|------|-----|------------|----------|----------|
| B0 | **0.5745** | 0.8511 | 0.4714 | 162 |
| B0-full | **0.0000** | 0.5035 | 0.6533 | 0 |
| B2 | 0.4326 | 0.6383 | 0.4436 | 122 |
| B3 | 0.4078 | 0.5213 | 0.4355 | 115 |
| B4 | 0.4078 | 0.5213 | 0.4355 | 115 |
| **B5** | **0.0000** | **0.5035** | **0.6533** | **0** |

**解讀（本 run）**

- **B0** 顯示「無 grounded 約束的樂觀判斷」FAR 約 **57%**，適合作為反面教材。
- **B5 與 B0-full** 在 FAR／resolution／Macro-F1 **完全一致**：CORG 判斷面採 **嚴格義務閉包**，安全面與保守基線同級，但 **B5 同時參與 release 實驗**（見下）。
- **B4→B5**：FAR 自 **40.78% → 0%**（見 Table 6）。
- **B3 vs B2**：resolution **下降** 約 11.7 個百分點（與「graph 必提升 resolution」假設不一致，論文需討論或改敘事）。

### 7.3 Table 5 — 證據釋出（counterevidence mutation）

| 綁定策略 | 未授權釋出率 | 無關仍釋出率 | 相關 n | 無關 n |
|----------|--------------|--------------|--------|--------|
| AGT-cited | **1.00** | 1.00 | 36 | 36 |
| AGT-global | 0.00 | 1.00 | 36 | 36 |
| **AGT+CORG** | **0.00** | 1.00 | 36 | 36 |
| AGT+same_R | 0.00 | 1.00 | 36 | 36 |

**解讀**：**AGT-cited vs AGT+CORG** 在未授權釋出上形成 **100% vs 0%** 的強對照，支持 CORG 作為 **release gate** 的敘事。  
**Irrelevant still release** 各策略均為 1.0，表示 **本 pilot 未在「無關證據過濾」上拉開差距**。

### 7.4 Table 6 — 關鍵對照與 Bootstrap 95% CI

| 對照 | 指標 | Δ | 95% CI（若適用） |
|------|------|---|------------------|
| B5 vs B4 | FAR | **−0.4078** | [−0.4545, −0.3601] |
| B3 vs B2 | resolution | −0.1170 | [−0.1625, −0.0742] |
| AGT+CORG vs AGT-cited | unauthorized_release_rate | +1.0（表方向：CORG 多擋下） | — |

### 7.5 消融（Ablation，n=80 子樣本）

| 設定 | FAR（full） | FAR（ablated） |
|------|-------------|----------------|
| disable_cited_span | 0.0 | 0.0 |
| ignore_coverage_flag | 0.0 | 0.0 |

在 B5 已 FAR=0 之前提下，子樣本消融 **未顯示 FAR 差異**（可能飽和）。

### 7.6 全 bundle 其他結果摘要

**標註（模擬）**

- Overall Cohen's κ（A vs B）：**0.8853**
- Raw agreement：**0.9336**
- 註：B 為 gold + **7% 標籤噪聲**，**非專家實標**。

**公開探針（pilot，非官方 split）**

| 任務 | n | 指標 | 值 |
|------|---|------|-----|
| FEVEROUS_pilot | 60 | claim_verification_accuracy | 0.7333 |
| ContractNLI_pilot | 80 | label_accuracy_vs_gold | 0.7250 |
| ACE_pilot | 115 clauses | multi_clause_consistency_proxy | 0.6522 |
| RAGTruth_pilot | 12 | grounding_pass_rate | 1.0000 |

**注入／替換（proxy）**

| 項目 | 值 |
|------|-----|
| AgentDojo 風格 injection_success | B2 **0.833** / B4 **0.833** / B5 **0.667** |
| Loopjacking 風格 unchanged_utility_B5 | **0.167**（乾淨 SUPPORTED 仍維持 SUPPORTED 之比例） |

**跨域（合成水＋電，各 30 包）**

| 域 | 方法 | FAR | Resolution |
|----|------|-----|------------|
| pooled | B4 | 0.40 | 0.833 |
| pooled | B5 | **0.00** | **0.00** |

B5 在跨域上 **FAR 仍為 0**，但 **resolution 為 0** → 不宜宣稱跨域「又準又全」，僅能強調 **保守拒判**。

**延遲（200 包，規則管線）**

| 方法 | p50 (ms/包) |
|------|-------------|
| B0 | 0.0005 |
| B5 | 0.0020 |
| process RSS | ~33.5 MB |

---

## 八、圖表與論文產物

| 檔案 | 說明 |
|------|------|
| `data/eval_sac/figures/figure1_architecture.png` | 系統／評估架構圖 |
| `data/eval_sac/figures/figure2_corg_release.png` | CORG 釋出流程 |
| `data/eval_sac/figures/risk_resolution_curve.png` | 風險—解析權衡曲線 |
| `data/eval_sac/figures/latency_p50.png` | 延遲比較 |
| `docs/Semi-Shield_SAC27_Empirical_Evaluation_Paper.pdf` | 填表後 measured 稿（合併協議 PDF） |
| `docs/文件/Semi-Shield_ISMS_CORG實驗評估論文章節.md` | 中文論文章節 |
| `docs/SAC_EVAL_RESULTS.md` | 英文精簡結果 |

---

## 九、限制與誠實聲明（撰寫論文必放）

1. **Gold 標籤** 主要來自 **ot/ 語料 + rubric 合成情境**，不是獨立專家盲標。  
2. **Cohen's κ** 為 **模擬雙標者**（含人工噪聲），不能替代 IRR 專家實驗。  
3. **公開探針** 為 **本地 pilot**，未使用 FEVEROUS / ContractNLI 等 **官方 test split**。  
4. **跨域** 為 **合成模板**，非真實水務／電力 OT 場域。  
5. **B2–B5 判斷管線** 在 eval 中為 **可重現規則實作**，與 production **Ollama 全文** 有差距；`ollama_b2_writer` 需另跑。  
6. **延遲** 未含 GPU LLM 推理；VRAM 需外部 `nvidia-smi`。  
7. **Irrelevant release** 指標目前 **無區分度**，不宜過度宣稱。

---

## 十、如何重現實驗

在專案根目錄：

```bat
:: 核心 Table 2/4/5/6（120 groups → 522 packets，--groups 20）
py -3.12 scripts\run_sac_corg_evaluation.py --groups 20

:: 全 bundle + 圖（較久）
py -3.12 scripts\run_sac_full_data_generation.py --groups 20

:: 資料總覽 HTML / Markdown
py -3.12 scripts\build_data_overview.py

:: 啟動平台（對話、戰情）
run_ollama.bat
:: 或 .venv\Scripts\python.exe app.py
```

輸出目錄：`data/eval_sac/`。

---

## 十一、相關程式與文件索引

| 類別 | 路徑 |
|------|------|
| 評估核心 | `code/eval/`（`scenarios.py`, `baselines.py`, `metrics.py`, `release_eval.py`, `obligations.py`） |
| Flask 主程式 | `code/server/app.py` |
| 架構說明 | `docs/文件/ARCHITECTURE.md` |
| 目錄約定 | `docs/文件/PROJECT_LAYOUT.md` |
| 合規矩陣 | `docs/文件/COMPLIANCE_MATRIX.md` |
| 證據鏈論文 | `docs/文件/Semi-Shield_AI_Evidence_Chain_論文版.md` |
| 實驗結果 JSON | `data/eval_sac/results.json` |
| 完整 bundle | `data/eval_sac/full_data_bundle.json` |
| 可讀儀表板 | `data/data_overview.html`、`data/資料總覽.md` |

---

## 十二、一頁結論（給簡報用）

**專案**：Semi-Shield 把 OT syslog 與 ISMS 控制項變成 **可追溯證據鏈 + 覆核佇列 + 地端 AI 診斷**，並以 **CORG** 約束「能說合規」與「能釋出證據」。  

**實驗**：在 **522** 個合成 held-out 封包、**6** 類 OT 控制項情境上，**B5（CORG）** 將 **FAR 從 40.8%（B4）降到 0%**，並使 **未授權證據釋出（AGT-cited 100% → AGT+CORG 0%）**。  

**代價／限制**：判斷面與 **B0-full** 同級保守；graph 檢索未提升 resolution；跨域與 LLM writer 仍待加強；gold 與 κ 需專家化後方可對外投稿強宣稱。

---

*文件結束。若數值更新，請重跑 `run_sac_corg_evaluation.py` 後以 `build_data_overview.py` 刷新 `data/資料總覽.md`。*
