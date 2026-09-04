#!/usr/bin/env python3
"""CLI：跨平台輔助（供 .bat / .sh 呼叫）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.platform_compat import find_venv_python, kill_listeners_on_port  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("Usage: platform_utils.py kill-port [PORT] | python-path", file=sys.stderr)
        return 2
    cmd = args[0]
    if cmd == "kill-port":
        port = int(args[1] if len(args) > 1 else __import__("os").environ.get("PORT", "2000"))
        killed = kill_listeners_on_port(port)
        return 0 if killed or True else 0
    if cmd == "python-path":
        print(find_venv_python(ROOT))
        return 0
    print(f"Unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
