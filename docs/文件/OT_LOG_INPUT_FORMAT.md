# OT 日誌輸入格式

監控戰情室與 AI 對話共用 `ot/` 目錄掃描，支援以下格式：

## 1. TXT（Cisco flash / syslog 純文字）

- 路徑：`ot/**/*.txt`
- 每行一則 Cisco syslog（含 `%FACILITY-SEVERITY-MNEMONIC:`）
- 檔名建議：`C9300-48p_192.168.3.254_flash.txt`

## 2. JSONL（結構化 syslog）

- 路徑：`ot/**/*.jsonl`
- 支援格式：
  - **NDJSON**：每行一個 JSON 物件（建議）
  - **JSON 陣列**：`[{...}, {...}]` 整檔一個陣列
  - **Pretty-print**：多行縮排物件
  - **包裝物件**：根物件內 `events` / `records` / `logs` 陣列
- `message` 欄位需含 Cisco syslog 原文（與 TXT 相同分類規則）

### 建議欄位

| 欄位 | 必填 | 說明 |
|------|------|------|
| `message` | 是 | Cisco syslog 原文 |
| `event_ts` | 否 | ISO 8601 時間（覆寫顯示時間） |
| `severity` | 否 | info / warning / error / critical 等 |
| `canonical_device_id` | 否 | 設備 ID |
| `source_ip` | 否 | 來源 IP |
| `source_device_id` | 否 | 來源設備別名 |
| `original_id` | 否 | 原始序號（保留於事件） |
| `source_id` | 否 | 資料來源標記（如 syslog_norm） |

### 範例

```json
{"original_id":191,"canonical_device_id":"C9300-24P-F11C-A-3","source_device_id":"C9300-24P-A-1","event_ts":"2026-08-04T12:07:58+08:00","severity":"info","source_ip":"192.168.3.3","message":"<187>41170990: Aug  4 12:07:41 UTC: %IOSXE-3-PLATFORM: Switch 1 R0/0: kernel: sd 0:0:0:0: [sda] tag#0 device offline or changed","source_id":"syslog_norm"}
```

### 相容別名

- 訊息：`message` / `raw` / `log` / `syslog`
- 時間：`event_ts` / `timestamp` / `time` / `ts`
- 設備：`canonical_device_id` / `device` / `device_id` / `source_device_id`
- IP：`source_ip` / `ip` / `host_ip`

## 掃描行為

- `.txt` 與 `.jsonl` **合併計數**，進同一套六控制項 KPI
- 檔案未變更時使用快取，避免重複全量掃描
- 排除：`requirements.txt`、`readme.txt`、`evidence_registry.jsonl` 等非 OT 日誌檔
