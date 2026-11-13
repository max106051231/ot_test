# 專案目錄結構

Semi-Shield ISMS 根目錄整理後的約定如下。

```
ot_test/
├── app.py                 # 啟動入口（轉呼叫 code.server）
├── run_ollama.bat         # Ollama 模式一鍵啟動
├── run_edge.bat           # Edge / CPU 模式
├── setup_python312.bat    # Python 3.12 虛擬環境
├── import_ollama_models.bat
│
├── code/                  # 後端 Python 程式
│   ├── paths.py           # 專案路徑（project_root、web、config…）
│   ├── services/          # 可重用服務模組
│   │   ├── ollama_service.py
│   │   ├── rag_service.py
│   │   ├── guardrail_service.py
│   │   ├── compliance_service.py
│   │   ├── evidence_service.py
│   │   ├── review_queue.py
│   │   └── agent_orchestrator.py
│   └── server/
│       └── app.py         # Flask 主程式與 API 路由
│
├── web/                   # 前端 HTML 頁面
│   ├── platform.html
│   ├── agent_chat.html
│   ├── OT.html
│   └── compliance.html
│
├── config/                # 執行期設定
│   └── ollama_models.json
│
├── static/                # 靜態資源（/static）
├── compliance/            # ISO 27001 控制項 JSON
├── data/                  # evidence / review 執行期資料
├── ot/                    # OT 設備 syslog 樣本
├── docs/                  # 文件與簡報素材
├── scripts/               # 維運腳本（Ollama 匯入、vendor JS…）
├── train_ai/              # 模型訓練與 RAG 索引
└── _archive/              # 備份與暫存（不納入版控）
```

## 啟動

```bat
run_ollama.bat
```

或：

```bat
.venv\Scripts\python.exe app.py
```

## 注意

- 請一律從**專案根目錄**執行，路徑由 `code/paths.py` 解析。
- 新增服務請放在 `code/services/`，路由與 Flask 設定放在 `code/server/app.py`（後續可再拆成 `routes/` 子模組）。
