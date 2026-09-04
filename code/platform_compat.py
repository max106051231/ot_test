"""Linux / Windows / macOS 共用：Python 路徑、埠號清理、網路提示。"""
from __future__ import annotations

import os
import re
import signal
import socket
import subprocess
import sys
from pathlib import Path

from code.paths import project_root


def find_venv_python(root: Path | None = None) -> Path:
    """優先專案 .venv，其次 PATH 上的 python3 / python。"""
    root = root or project_root()
    if sys.platform == "win32":
        candidates = [
            root / ".venv" / "Scripts" / "python.exe",
            root / ".venv" / "Scripts" / "python3.exe",
        ]
    else:
        candidates = [
            root / ".venv" / "bin" / "python3",
            root / ".venv" / "bin" / "python",
        ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    for name in ("python3", "python"):
        found = __import__("shutil").which(name)
        if found:
            return Path(found).resolve()
    return Path(sys.executable).resolve()


def _pids_listening_on_port(port: int) -> list[int]:
    pids: set[int] = set()
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(
                ["netstat", "-ano"],
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            return []
        needle = f":{port} "
        for line in out.splitlines():
            if "LISTENING" not in line.upper() or needle not in line:
                continue
            parts = line.split()
            if parts:
                try:
                    pids.add(int(parts[-1]))
                except ValueError:
                    continue
        return sorted(pids)

    # Linux / macOS：lsof → ss
    for cmd in (
        ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
        ["lsof", "-ti", f":{port}"],
    ):
        try:
            out = subprocess.check_output(cmd, text=True, errors="ignore").strip()
            for tok in out.split():
                try:
                    pids.add(int(tok))
                except ValueError:
                    pass
            if pids:
                return sorted(pids)
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    try:
        out = subprocess.check_output(
            ["ss", "-ltnp", f"sport = :{port}"],
            text=True,
            errors="ignore",
        )
        for m in re.finditer(r"pid=(\d+)", out):
            pids.add(int(m.group(1)))
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return sorted(pids)


def kill_listeners_on_port(port: int, *, exclude_pid: int | None = None) -> list[int]:
    """
    關閉佔用 port 的舊行程（避免 Flask 缺路由）。
    設 KILL_STALE_PORT=0 可停用。
    """
    if os.environ.get("KILL_STALE_PORT", "1").strip().lower() in ("0", "false", "no"):
        return []
    me = exclude_pid if exclude_pid is not None else os.getpid()
    killed: list[int] = []
    for pid in _pids_listening_on_port(port):
        if pid <= 0 or pid == me:
            continue
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    check=False,
                    capture_output=True,
                )
            else:
                os.kill(pid, signal.SIGTERM)
            killed.append(pid)
            print(f"[Port {port}] stopped stale PID {pid}")
        except (ProcessLookupError, PermissionError, OSError) as e:
            print(f"[Port {port}] could not stop PID {pid}: {e}")
    return killed


def collect_lan_ipv4() -> list[str]:
    """補充區網 IPv4（Linux ip/hostname -I；Windows ipconfig）。"""
    addrs: list[str] = []

    def _add(ip: str) -> None:
        if not ip or ip.startswith("127.") or ip.startswith("169.254."):
            return
        if ip not in addrs:
            addrs.append(ip)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            _add(s.getsockname()[0])
    except OSError:
        pass

    if sys.platform == "win32":
        try:
            out = subprocess.check_output(
                ["ipconfig"], text=True, encoding="utf-8", errors="ignore"
            )
            for line in out.splitlines():
                if "IPv4" in line or "IP Address" in line:
                    part = line.split(":")[-1].strip()
                    _add(part.split("(")[0].strip())
        except Exception:
            pass
    else:
        try:
            out = subprocess.check_output(
                ["ip", "-4", "addr"], text=True, errors="ignore"
            )
            for m in re.finditer(r"\binet (\d+\.\d+\.\d+\.\d+)", out):
                _add(m.group(1))
        except Exception:
            try:
                out = subprocess.check_output(
                    ["hostname", "-I"], text=True, errors="ignore"
                )
                for tok in out.split():
                    _add(tok.strip())
            except Exception:
                pass

    addrs.sort(key=lambda ip: (0 if ip.startswith("26.") else (1 if ip.startswith("25.") else 2), ip))
    return addrs


def firewall_hint(port: int) -> str:
    if sys.platform == "win32":
        return f"（若連不上：Windows 防火牆需允許 Python／埠 {port} 傳入）"
    if sys.platform == "darwin":
        return f"（若連不上：macOS 防火牆需允許 Python／埠 {port}）"
    return (
        f"（若連不上：Linux 請開放埠 {port}，例如 "
        f"`sudo ufw allow {port}/tcp` 或 firewalld）"
    )


def ip_lookup_hint() -> str:
    if sys.platform == "win32":
        return "ipconfig"
    return "ip addr 或 hostname -I"
