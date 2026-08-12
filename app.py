"""
Semi-Shield ISMS — 啟動入口。

請從專案根目錄執行：python app.py  或  run_ollama.bat
"""
from code.server.app import app, run_server

__all__ = ["app", "run_server"]

if __name__ == "__main__":
    run_server()
