# Semi-Shield ISMS

OT 工控資安合規平台（ISO 27001、RAG、護欄、合規管線）。支援 **Windows** 與 **Linux / macOS**。

## 需求

- Python **3.12+**（建議 3.12；Ollama 模式可用 3.12–3.14）
- [Ollama](https://ollama.com/download)（預設後端）
- 可選：NVIDIA GPU（Transformers 本地模式）

## 快速開始

### Windows

```bat
REM 1. 建立虛擬環境（首次）
setup_python312.bat

REM 2. 安裝 Ollama 並匯入微調模型（首次）
import_ollama_models.bat

REM 3. 啟動後端
run_ollama.bat
```

瀏覽器開啟：http://127.0.0.1:2000/

### Linux / macOS

```bash
# 1. 建立虛擬環境（首次）
chmod +x scripts/setup_venv.sh run_ollama.sh import_ollama_models.sh
bash scripts/setup_venv.sh

# 2. 安裝 Ollama 並匯入模型（首次）
./import_ollama_models.sh

# 3. 啟動（另開終端機先執行 ollama serve，若尚未啟動）
./run_ollama.sh
```

瀏覽器開啟：http://127.0.0.1:2000/

### 通用（兩平台皆可）

```bash
# 在專案根目錄
python app.py          # Windows: py -3.12 app.py
# 環境變數與 .bat/.sh 相同，見下方
```

## 啟動腳本對照

| 用途 | Windows | Linux / macOS |
|------|---------|---------------|
| Ollama 後端（預設） | `run_ollama.bat` | `./run_ollama.sh` |
| Edge / CPU 模式 | `run_edge.bat` | `./run_edge.sh` |
| 匯入 Ollama 模型 | `import_ollama_models.bat` | `./import_ollama_models.sh` |
| 建立 venv | `setup_python312.bat` | `bash scripts/setup_venv.sh` |

## 常用環境變數

| 變數 | 預設 | 說明 |
|------|------|------|
| `PORT` | `2000` | Web 埠號 |
| `LLM_BACKEND` | `ollama` | `ollama` 或本地 Transformers |
| `OLLAMA_MODEL` | `qwen2.5:3b` | 預設 Ollama 模型 |
| `OT_ENABLE_RAG` | `1` | 知識庫 RAG |
| `ENABLE_GUARDRAIL` | `1` | 輸入護欄 |
| `KILL_STALE_PORT` | `1` | 啟動前關閉佔用 PORT 的舊行程 |

## 目錄結構

```
ot_test/
├── app.py                 # 啟動入口
├── run_ollama.bat / .sh   # 主啟動腳本
├── code/                  # 後端 Python
├── web/                   # 前端 HTML
├── ot/                    # OT 日誌樣本（可自動建立）
├── config/                # Ollama 模型別名等
└── scripts/               # 測試與工具
```

## 測試

```bash
python scripts/full_project_test.py
python scripts/test_gemma_llama_compare.py
```

報告：`docs/TEST_REPORT.md`、`docs/GEMMA_LLAMA_COMPARE_REPORT.md`

## 防火牆

- **Windows**：允許 Python 或埠 2000 傳入（或執行 `open_firewall_2000.bat`）
- **Linux**：例如 `sudo ufw allow 2000/tcp`
- **macOS**：系統設定 → 防火牆 → 允許 Python

## 疑難排解

1. **API 404 / 舊路由**：重啟 `run_ollama.bat` 或 `./run_ollama.sh`（會自動清理佔用埠號）
2. **Ollama 未連線**：先執行 `ollama serve` 或開啟 Ollama Desktop
3. **Linux 缺 lsof**：`sudo apt install lsof`（埠號清理用；亦可設 `KILL_STALE_PORT=0`）

## 授權

依專案原有授權；模型權重請遵循各 upstream 授權條款。
