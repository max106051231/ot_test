"""專案根目錄與常用路徑（所有模組統一由此解析）。"""
from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_root() -> Path:
    return _PROJECT_ROOT


def web_dir() -> Path:
    return _PROJECT_ROOT / "web"


def config_dir() -> Path:
    return _PROJECT_ROOT / "config"


def static_dir() -> Path:
    return _PROJECT_ROOT / "static"


def data_dir() -> Path:
    return _PROJECT_ROOT / "data"


def compliance_dir() -> Path:
    return _PROJECT_ROOT / "compliance"


def ot_logs_dir() -> Path:
    ot = _PROJECT_ROOT / "ot"
    ot_upper = _PROJECT_ROOT / "OT"
    if ot.is_dir():
        return ot
    if ot_upper.is_dir():
        return ot_upper
    return ot


def train_ai_dir() -> Path:
    return _PROJECT_ROOT / "train_ai"
