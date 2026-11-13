from __future__ import annotations

import re
from pathlib import Path

from code.eval.sac_types import EvidenceRecord
from code.paths import ot_logs_dir

_RULES: list[tuple[str, re.Pattern]] = [
    ("access_control", re.compile(r"SEC_LOGIN|LOGIN|AAA|RADIUS|TACACS|SSH", re.I)),
    ("supplier_security", re.compile(r"CDP|LLDP|NEIGHBOR|REMOTE", re.I)),
    ("malware_defense", re.compile(r"IPS|MALWARE|THREAT|ATTACK|HOST", re.I)),
    ("patch_management", re.compile(r"IMAGE|SOFTWARE|UPGRADE|SMU|IOS|VERSION", re.I)),
    ("recipe_audit", re.compile(r"CONFIG_I|INTERFACE|PORT-SECURITY|CHANGED", re.I)),
    ("sec_gem_log", re.compile(r"SNMP|CRYPTO|CERTIFICATE|TLS|PKI", re.I)),
]

_KEY_TO_ANNEX = {
    "access_control": "A.5.15",
    "supplier_security": "A.5.19",
    "malware_defense": "A.8.7",
    "patch_management": "A.8.8",
    "recipe_audit": "A.8.19",
    "sec_gem_log": "A.8.24",
}


def classify_line(line: str) -> str | None:
    s = (line or "").strip()
    if len(s) < 10:
        return None
    for key, pat in _RULES:
        if pat.search(s):
            return key
    return None


def _severity(line: str) -> int:
    m = re.search(r"-(\d)-", line)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    if re.search(r"FAIL|DENIED|VIOLAT", line, re.I):
        return 3
    return 5


def load_ot_log_pool(max_lines: int = 8000) -> list[tuple[str, str, int]]:
    """(control_key, raw, severity) from ot/."""
    out: list[tuple[str, str, int]] = []
    root = ot_logs_dir()
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".txt", ".log", ".jsonl"):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if path.suffix.lower() == ".jsonl":
                import json

                try:
                    obj = json.loads(line)
                    line = (
                        obj.get("message")
                        or obj.get("msg")
                        or obj.get("raw")
                        or line
                    )
                except json.JSONDecodeError:
                    pass
            key = classify_line(str(line))
            if not key:
                continue
            out.append((key, str(line).strip()[:400], _severity(str(line))))
            if len(out) >= max_lines:
                return out
    return out


def make_record(
    idx: int,
    key: str,
    raw: str,
    severity: int,
    *,
    site: str,
    asset: str,
    trusted: bool = True,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=f"ev-{key}-{idx:05d}",
        raw=raw,
        site=site,
        asset=asset,
        control_key=key,
        severity=severity,
        trusted=trusted,
    )
