import json
import time
from openai import OpenAI

# 實例化 API Client
client = OpenAI(api_key="YOUR_API_KEY")

# 組合變數矩陣，可產生數千種不重複指令情境
TOPICS = [
    "A.5 組織控制 (A.5.7 威脅情報, A.5.23 雲端服務資安, A.5.30 營運持續)",
    "A.6 人員控制 (A.6.3 資安培訓, A.6.7 遠距工作, A.6.8 事件通報)",
    "A.7 實體控制 (A.7.4 實體監控, A.7.9 桌面淨空, A.7.10 儲存介質)",
    "A.8 技術控制 (A.8.8 漏洞管理, A.8.12 DLP, A.8.28 安全程式碼)",
    "OT/ICS 工控資安 (Modbus, S7comm, OPC UA, IEC 62443 與 ISO 27001 整合)",
    "Cloud & DevSecOps (AWS/Azure Log 診斷, K8s 審計, CI/CD 安全)"
]

SYSTEM_PROMPT = """你是一個精通 ISO/IEC 27001:2022 標準、NIST SP 800-53 與 OT/IT 資安稽核的專家。
請幫我編寫微調 LLM 用的問答資料對，格式必須嚴格符合 JSON 物件：
{
  "instruction": "題目說明 (包含明確情境、Log 診斷或標準條款實施問題)",
  "output": "詳細解答 (包含 1. 事件/條款解讀, 2. 不合規原因與 ISO 27001 條款對齊, 3. 具體修補與落地指南)"
}
請確保回答專業、精準、直接給出實質內容，不含任何開場白。"""

generated_data = []

for i in range(1, 125): # 執行 125 次，每次產出 8 筆，共約 1000 筆
    topic = TOPICS[i % len(TOPICS)]
    user_prompt = f"請針對領域【{topic}】，生成 8 筆高質量的微調問答對 JSON 陣列。請包含至少 3 筆真實系統 Log 分析（如 Syslog, AWS, K8s, FW Log）。"
    
    response = client.chat.completions.create(
        model="gpt-4o", # 或使用相應的模型名稱
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    # 提取回應並格式化存檔
    content = json.loads(response.choices[0].message.content)
    # 追加至本地 JSON 檔案...
    print(f"Batch {i} completed.")
    time.sleep(1)