# Ollama 後端設定

Semi-Shield 預設以 **Ollama** 管理 LLM（`LLM_BACKEND=ollama`）。Flask 不再直接載入 HuggingFace 權重，改呼叫本機 Ollama API。

## 1. 安裝 Ollama

- 下載：https://ollama.com/download
- 確認服務：`ollama serve`（Windows 安裝後通常常駐）

## 2. 拉取基底模型

```powershell
ollama pull qwen3:4b
ollama pull qwen2.5:3b
ollama pull gemma2:2b
```

## 3. 一鍵匯入專案模型（推薦）

專案已含微調權重與 preset 對照，執行：

```powershell
import_ollama_models.bat
```

或：

```powershell
python scripts/import_models_to_ollama.py
```

腳本會：

1. `ollama pull` 各 preset 對應基底（qwen3:4b、qwen2.5:3b、phi4、gemma2:2b 等）
2. 匯入 `train_ai/models/*/model.safetensors`（目前：`qwen3_4b_ot`、`gemma_2b_ot`）
3. 其餘 preset（尚無本地 merge）建立 **基底 + Semi-Shield SYSTEM** 別名

只匯入指定模型：

```powershell
python scripts/import_models_to_ollama.py --only qwen3_4b_ot,gemma_2b_ot
```

縮小模型体积（量化）：

```powershell
python scripts/import_models_to_ollama.py --quantize q4_K_M
```

## 4. 手動 Modelfile（進階）

若已有 merge 後的 Safetensors，可在模型目錄建立 `Modelfile.ollama`：

```dockerfile
FROM .
PARAMETER temperature 0
PARAMETER num_ctx 8192
SYSTEM 你是 Semi-Shield Cyber Agent，專精 OT 工控與 ISO 27001 合規。
```

```powershell
cd train_ai\models\qwen3_4b_ot
ollama create qwen3_4b_ot -f Modelfile.ollama
```

別名對照見 `config/ollama_models.json`。

## 5. 啟動平台

```powershell
set LLM_BACKEND=ollama
set OLLAMA_MODEL=qwen2.5:3b
set OLLAMA_BASE_URL=http://127.0.0.1:11434
set OT_ENABLE_RAG=1
python app.py
```

或使用 `run_ollama.bat`（已預設 `OT_ENABLE_RAG=1`）。

### Python 3.12（建議）

本機若預設為 Python 3.14，numpy／torch／護欄 ML 可能無法載入。請改用 3.12：

```powershell
setup_python312.bat
run_ollama.bat
```

腳本會建立 `.venv`（Python 3.12）並安裝 `requirements-ollama.txt` + CPU 版 torch。

## 6. RAG 知識庫（專業回答）

聊天會自動從 `train_ai/train_rag/knowledge_base.json`（681 筆 OT／ISO 27001 知識）檢索相關內容，注入 LLM prompt，讓回答更貼近顧問口吻。

- **預設已開啟**：`OT_ENABLE_RAG=1`
- **關閉**：`set OT_ENABLE_RAG=0`
- **每題都檢索**：`set OT_RAG_ALWAYS=1`
- 若 numpy／sentence-transformers 不可用（如 Python 3.14），會自動改用**關鍵字檢索**（狀態列顯示 `RAG ON(keyword·681)`）
- 有向量環境時會用 embedding 檢索（`RAG ON(vector·681)`）
- 回答下方可展開「知識庫引用」查看 RAG 來源

重建知識庫索引：

```powershell
cd train_ai\train_rag
python build_rag_from_current.py --rebuild-index
```

## 7. 切換回 HuggingFace 直載（舊行為）

```powershell
set LLM_BACKEND=transformers
python app.py
```

## 環境變數

| 變數 | 預設 | 說明 |
|------|------|------|
| `LLM_BACKEND` | `ollama` | `ollama` 或 `transformers` |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama API |
| `OLLAMA_MODEL` | `qwen2.5:3b` | 預設模型 tag |
| `OLLAMA_TIMEOUT` | `300` | 推論逾時（秒） |
| `OT_ENABLE_RAG` | `1` | 聊天 RAG 知識庫檢索 |
| `OT_RAG_ALWAYS` | `0` | `1` = 每題都跑 RAG |
