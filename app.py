"""
Semi-Shield ISMS — 啟動入口。

請從專案根目錄執行：
  Windows: run_ollama.bat  或  python app.py
  Linux/macOS: ./run_ollama.sh  或  python3 app.py
"""
from code.server.app import app, run_server

__all__ = ["app", "run_server"]

if __name__ == "__main__":
    run_server()
