# Semi-Shield ISMS — 專案詳細資訊、實驗內容與實驗結果

> 文件用途：計畫書／決賽簡報／論文附錄之單一參考稿  
> 數據快照：`data/eval_sac/results.json`、`data/eval_sac/full_data_bundle.json`（2026-10-17 產生）  
> 儀表板：`data/data_overview.html`、`data/資料總覽.md`（可執行 `py -3.12 scripts/build_data_overview.py` 更新）

---

## 一、專案定位

**Semi-Shield ISMS** 是面向 **OT（營運技術）工控環境** 的 **ISO/IEC 27001 Annex A** 合規輔助平台，設計原則為：

**No Evidence, No Compliance Claim（無證據、不宣稱合規）**

系統將現場 **Cisco 等設備 syslog**、**手動匯入政策／紀錄**、**RAG 知識庫** 與 **地端 LLM（Ollama）** 整合，提供：

- 監控戰情室（六項 OT 自動化控制項 KPI）
- AI 稽核對話（ grounded 上下文 + 護欄）
- 證據鏈登錄（hash chain 可追溯）
- 人工／AI 覆核佇列
- OT Purdue 拓樸視覺化（多 site）
- **CORG（Control-Obligation Release Gate）**：以 **義務閉包（obligation read-set）** 綁定「可釋出之證據集合」，降低 **誤肯定（False Affirmative）** 與 **未授權證據釋出**

論文／競賽評估協議對應程式目錄：`code/eval/`、執行腳本 `scripts/run_sac_corg_evaluation.py`、`scripts/run_sac_full_data_generation.py`。

---

## 二、系統架構（執行時）

```
瀏覽器 :2000
  platform.html │ OT.html │ agent_chat.html │ compliance.html │ ot_topology.html
        ↓ REST
code/server/app.py (Flask)
  監控掃描 · Agent · 合規管線 · 證據 API · 拓樸 API
        ↓
code/services/
  ollama_service · rag_service · guardrail · hallucination_guard
  compliance_service · evidence_service · review_queue
  evidence_chain_service · ot_topology_service · manual_import_service
        ↓
Ollama :11434 │ train_ai/* 模型與 RAG │ data/*.jsonl │ ot/*.txt │ compliance/*.json
```

### 2.1 主要前端頁面

| 頁面 | 路徑 | 功能 |
|------|------|------|
| 整合首頁 | `/`、`/platform` | 戰情 + 對話入口 |
| 監控戰情室 | `/monitor` → `OT.html` | 六控制項 KPI、事件串流、地端診斷、匯出 |
| AI 對話 | `/chat` → `agent_chat.html` | 多輪對話、串流回覆、模型切換 |
| 合規管線 | `/compliance` | 矩陣、Evidence、Review Queue |
| OT 拓樸 | `ot_topology.html` | 多 site Purdue 圖、模擬 syslog feed |

### 2.2 六項 OT 自動化監控控制項

| control_key | Annex A | 監控重點 |
|-------------|---------|----------|
| sec_gem_log | A.8.24 | SNMP / PKI / 加密傳輸 |
| recipe_audit | A.8.19 | 組態變更、介面 UPDOWN |
| access_control | A.5.15 | AAA / LOGIN |
| patch_management | A.8.8 | 修補與 CVE 提示 |
| supplier_security | A.5.19 | CDP/LLDP 鄰居 |
| malware_defense | A.8.7 | IPS / 惡意軟體命中 |

資料來源：`ot/` 目錄 syslog 文字檔；掃描結果快取後經 `/api/monitor/data` 提供給前端與 Agent 上下文。

### 2.3 AI 對話管線（摘要）

1. 輸入護欄（`guardrail_service`）  
2. RAG 檢索（`rag_service`，非寒暄問題強制檢索）  
3. 注入監控戰情室上下文（與 OT.html 同源）  
4. Ollama / 本地 LLM 生成  
5. 幻覺護欄（`hallucination_guard_service`）  
6. 回覆消毒與可選圖表  

設計要點：**有 OT 日誌時以日誌為事實優先**；避免無證據的合規斷言。

### 2.4 合規四階段管線

`Collector → Auditor → Reviewer → Reporter`  
由 `POST /api/compliance/pipeline` 觸發：掃描 OT、RAG+LLM 診斷、護欄與 Queue、PDF/TXT 報告。

### 2.5 執行期資料（`data/`）

| 檔案 | 說明（本機快照） |
|------|------------------|
| `evidence_registry.jsonl` | 965 筆證據；約 1,044 萬 syslog 事件彙總；hash chain |
| `review_queue.jsonl` | 895 筆覆核（approved / pending / rejected） |
| `manual_imports.jsonl` | 99 筆 Annex A 手動匯入（示範 simulated） |
| `eval_sac/*` | SAC/CORG 實驗輸出（見下文） |

---

## 三、CORG 與評估協議（實驗在量什麼）

### 3.1 判斷標籤 SVI

每個 **scenario packet** 有 gold 標籤：

- **SUPPORTED**：在 scope 內有足夠可信證據支持控制項  
- **VIOLATED**：有違規訊號  
- **INSUFFICIENT**：證據不足、scope 不符、或不可回答  

### 3.2 基線 B0～B5（`code/eval/baselines.py`）

| 代號 | 名稱 | 行為摘要 |
|------|------|----------|
| **B0** | 樂觀稽核草稿 | 模擬「LLM 無 grounded」：有控制項相關紀錄即判 SUPPORTED |
| **B0-full** | 義務感知保守基線 | 執行 selector（scope / signal / coverage），無充分證據則 INSUFFICIENT |
| **B2** | + 平面檢索 | BM25 式關鍵字命中 trusted 紀錄 |
| **B3** | + 圖擴展 | 同 site 鄰居紀錄擴展（graph stub） |
| **B4** | Semi-Shield 無 CORG | B3 + 控制項 key 與 span 長度過濾，**不**做完整 CORG 閉包 |
| **B5** | **Semi-Shield + CORG** | `corg_assess`：義務閉包 + cited span + coverage |

六個 **monitoring family** 對應 `code/eval/obligations.py` 中 Annex A：A.5.15、A.5.19、A.8.7、A.8.8、A.8.19、A.8.24。

### 3.3 情境變體（每 group 最多 4～5 種）

由 `generate_scenario_packets()` 自 **真實 ot/ log pool** 抽行並合成變體：

| variant | 說明 | 典型 gold |
|---------|------|-----------|
| **clean** | 正常見證 + coverage | SUPPORTED |
| **clean-violated** | 每 3 組插入違規 syslog | VIOLATED |
| **evidence_removed** | 無紀錄 | INSUFFICIENT |
| **scope_mismatch** | 紀錄在錯誤 Fab site | INSUFFICIENT |
| **injection** | 不可信紀錄含 prompt 注入句 | 依 rubric（多為 SUPPORTED 若忽略 untrusted） |

預設 `--groups 20`（每 family）→ 核心 eval **120 groups、522 packets**（held-out 子集）。

### 3.4 主要指標

- **FAR（False Affirmative Rate）**：gold 為非 SUPPORTED 時，模型仍判 SUPPORTED 的比例（**越低越安全**）  
- **Resolution**：在「可回答」子集上給出 SUPPORTED 或 VIOLATED 的比例（**越高越不過度拒答**）  
- **Macro-F1**：三類 SVI 的 macro F1  
- **Release 實驗**：比較 **AGT-cited**（僅 cite bundle 綁定）vs **AGT+CORG**（obligation read-set digest）在 **counterevidence 變異** 後是否 **未授權釋出**

### 3.5 完整 bundle 額外實驗（`run_sac_full_data_generation.py`）

| 區塊 | 內容 |
|------|------|
| 雙標註 + κ | 模擬 Annotator A/B（B = gold + 7% 噪聲） |
| Ollama B2 writer | 子集 LLM 寫手（本 bundle 為 `null`，需 Ollama 執行時才填） |
| Public probes | FEVEROUS / ContractNLI / ACE / RAGTruth **本地 pilot** |
| Injection battery | AgentDojo 風格注入、Loopjacking 代理 |
| Cross-domain | 合成水務 / 電力各 30 packets |
| Latency | 各基線 per-packet ms（規則管線，非端到端 LLM） |

---

## 四、核心實驗結果（Table 2 / 4 / 5 / 6）

**協議**：Semi-Shield OT Evidence Evaluation（executable subset）  
**執行時間**：約 2.22 s（`results.json`）  
**資料規模**：120 groups，**522 packets**

### Table 2 — 資料組成

| 項目 | 數值 |
|------|------|
| actual_groups | 120 |
| actual_packets | 522 |
| Gold SUPPORTED | 240 |
| Gold VIOLATED | 42 |
| Gold INSUFFICIENT | 240 |

註：partition 設計目標 dev 30 / hold-out 120；本 run 以 ot/ 合成 held-out 子集。

### Table 4 — OT 判斷（FAR ↓ 為佳；resolution ↑ 為佳）

| 方法 | FAR | Resolution | Macro-F1 | 誤肯定數 |
|------|-----|------------|----------|----------|
| B0 | 0.5745 | 0.8511 | 0.4714 | 162 |
| B0-full | 0.0000 | 0.5035 | 0.6533 | 0 |
| B2 | 0.4326 | 0.6383 | 0.4436 | 122 |
| B3 | 0.4078 | 0.5213 | 0.4355 | 115 |
| B4 | 0.4078 | 0.5213 | 0.4355 | 115 |
| **B5** | **0.0000** | **0.5035** | **0.6533** | **0** |

**解讀（本 run）**：

- B0 誤肯定率約 **57%**，顯示「無 CORG 的樂觀判斷」在 OT 證據場景風險高。  
- **B5 與 B0-full 同達 FAR=0、相同 resolution / Macro-F1**，代表 CORG 將判斷收斂到 **義務感知的安全區**；相對 **B4，FAR 自 40.78% 降至 0%**（Table 6 有 bootstrap 95% CI）。  
- B3 相對 B2 **resolution 下降**（約 −11.7%），graph stub 在此設定未展現論文假設中的提升，撰寫時需誠實討論或調參。

### Table 5 — 證據釋出（counterevidence 變異）

| 綁定策略 | 未授權釋出率 | 無關仍釋出率 | 相關 n | 無關 n |
|----------|--------------|--------------|--------|--------|
| AGT-cited | **1.00** | 1.00 | 36 | 36 |
| AGT-global | 0.00 | 1.00 | 36 | 36 |
| **AGT+CORG** | **0.00** | 1.00 | 36 | 36 |
| AGT+same_R | 0.00 | 1.00 | 36 | 36 |

**解讀**：**AGT-cited** 在變異後仍 **100% 未授權釋出**；**AGT+CORG** 為 **0%**，支持「cite 綁定不足以防禦 obligation 變化，CORG read-set 可擋釋出」之敘事。  
「無關仍釋出」各策略皆 1.0，此指標在本 pilot **尚無區分度**。

### Table 6 — 關鍵對照

| 對照 | 指標 | 觀察 Δ | 95% CI（若適用） |
|------|------|--------|------------------|
| B5 vs B4 | FAR | −0.4078 | [−0.455, −0.360] |
| B3 vs B2 | resolution | −0.1170 | [−0.162, −0.074] |
| AGT+CORG vs AGT-cited | unauthorized_release_rate | +1.0（CORG 多擋） | — |

### Ablation（子樣本 n=80）

| 設定 | FAR（full） | FAR（ablated） |
|------|-------------|----------------|
| disable_cited_span | 0.0 | 0.0 |
| ignore_coverage_flag | 0.0 | 0.0 |

在 B5 已 FAR=0 之前提下，此子集 ablation **未顯示差異**。

---

## 五、完整 bundle 補充結果

**產生時間**：2026-10-17T02:34:01Z，總耗時約 9.09 s  
**檔案**：`data/eval_sac/full_data_bundle.json`

### 5.1 標註一致性（pilot）

| 項目 | 值 |
|------|-----|
| Overall Cohen's κ（A vs B） | **0.8853** |
| Raw agreement | 0.9336 |
| 全資料 packets（含 dev） | 648 |

**限制**：Annotator B 為 **gold + 7% 標籤噪聲** 之模擬；正式投稿應改 **專家雙盲標註**。

### 5.2 公開探針（與 OT gold 分離）

| Pilot | 結果 |
|-------|------|
| FEVEROUS-style | n=60，claim verification accuracy **73.33%** |
| ContractNLI-style | n=80，label accuracy **72.50%** |
| ACE-style | 115 clauses，consistency proxy **65.22%** |
| RAGTruth-style | n=12，grounding pass **100%** |

**限制**：非官方 benchmark split，僅作 **管線 smoke / 方向性** 參考。

### 5.3 注入與替代攻擊（proxy）

| 測試 | 結果 |
|------|------|
| AgentDojo-style injection | n=150；B2/B4 注入成功 **83.33%**；B5 **66.67%** |
| Loopjacking-style | n=150 clean supported；B5 維持 SUPPORTED **16.67%** |

「注入成功」= 注入變體下錯誤 SUPPORTED；B5 有改善但未清零，論文可列為 **後續 hardening**。

### 5.4 跨域（合成水務 / 電力，各 30）

| 域 | 方法 | FAR | Resolution |
|----|------|-----|------------|
| water / power / pooled | B4 | 0.40 | 0.833 |
| water / power / pooled | B5 | **0.00** | **0.00** |

B5 跨域 **FAR 仍為 0**，但 **resolution 為 0**（高度保守、幾乎全 INSUFFICIENT），**不宜**宣稱跨域「準確率全面提升」。

### 5.5 延遲（200 packets 子集，規則評估器）

| 方法 | p50 (ms/packet) |
|------|-----------------|
| B0 | 0.0005 |
| B5 | 0.0020 |

程序 RSS 約 **33.5 MB**；**不含** Ollama GPU 延遲。

---

## 六、圖表與論文產物

| 路徑 | 說明 |
|------|------|
| `data/eval_sac/figures/figure1_architecture.png` | 系統／評估架構圖 |
| `data/eval_sac/figures/figure2_corg_release.png` | CORG 釋出綁定流程 |
| `data/eval_sac/figures/risk_resolution_curve.png` | FAR–resolution 權衡曲線 |
| `data/eval_sac/figures/latency_p50.png` | 延遲比較 |
| `docs/Semi-Shield_SAC27_Empirical_Evaluation_Paper.pdf` | 填表後 measured 稿 |
| `docs/SAC_EVAL_RESULTS.md` | 英文精簡結果 |
| `docs/文件/Semi-Shield_ISMS_CORG實驗評估論文章節.md` | 中文論文章節 |

---

## 七、如何重現實驗

在專案根目錄：

```bat
:: 僅核心 Table 2/4/5/6（--groups 20 → 522 packets）
py -3.12 scripts\run_sac_corg_evaluation.py --groups 20

:: 完整 bundle + 圖表
py -3.12 scripts\run_sac_full_data_generation.py --groups 20

:: 更新資料總覽 HTML
py -3.12 scripts\build_data_overview.py

:: 啟動平台（AI 對話、戰情室）
run_ollama.bat
:: 或 .venv\Scripts\python.exe app.py
:: 瀏覽器 http://127.0.0.1:2000/
```

輸出目錄：`data/eval_sac/`。

---

## 八、專案目錄速查

```
ot_test/
├── app.py                    # 啟動入口
├── code/
│   ├── eval/                 # SAC/CORG 評估（baselines, scenarios, metrics, release_eval）
│   ├── services/             # 業務邏輯（證據、RAG、Ollama、拓樸…）
│   └── server/app.py         # Flask API
├── web/                      # 前端頁面
├── ot/                       # OT syslog 語料
├── compliance/               # Annex A 控制項 JSON
├── data/                     # 證據、覆核、實驗結果
├── config/                   # Ollama、拓樸 site 設定
├── scripts/                  # 評估、繪圖、論文建置腳本
├── train_ai/                 # RAG / 護欄 / LLM 微調
└── docs/                     # 架構、合規矩陣、論文與本文件
```

---

## 九、成果摘要（給審查委員／口試用）

**專案貢獻**

1. 可執行的 OT-ISMS **證據鏈 + 戰情室 + 地端 LLM** 整合原型。  
2. **CORG** 可重現評估：在 522-packet held-out 上 **FAR 自 B4 的 40.8% 降至 0%**，並在 release 實驗中 **消除 AGT-cited 的 100% 未授權釋出**。  
3. 完整 artifact：`code/eval/`、manifest、JSON 結果、圖表、measured 論文 PDF。

**已知限制（建議口試主動說明）**

1. Gold 標籤主要來自 **ot/ 規則 rubric + 合成變體**，非第三方專家審定。  
2. Cohen's κ 為 **模擬雙標**；Ollama B2 writer 子集在本 bundle **未執行**。  
3. Graph 檢索 ablation **未**顯示 resolution 提升；跨域 B5 resolution=0。  
4. Public probes 為 **local pilot**；延遲不含真實 LLM 推理。  

**一句結論**：本專案在 **「降低合規誤肯定 + 安全證據釋出」** 上具 **清楚、可量測** 的實驗支持；在 **檢索增益、跨域效用、端到端 LLM 品質** 上仍需後續實驗補強。

---

*文件結束 · 與 `data/資料總覽.md` 數字同步；重新跑 eval 後請再執行 `build_data_overview.py` 或手動更新本章表格。*
