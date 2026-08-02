"""
依現有 Cisco flash .txt 日誌格式，擴充事件至合計約 50,000 筆。
用法（在專案根目錄或 ot/ 下）：
  python ot/expand_logs_to_50k.py
"""
from __future__ import annotations

import random
import re
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET_TOTAL = 50_000
EVENT_RE = re.compile(
    r"^([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}(?:\s+[A-Z]+)?)\s*:\s*"
    r"%([A-Z0-9_]+)-(\d+)-([A-Z0-9_]+):\s*(.*)$"
)

TEMPLATES = [
    # recipe_audit / link
    ("LINK", 3, "UPDOWN", "Interface GigabitEthernet1/0/{port}, changed state to {state}"),
    ("LINEPROTO", 5, "UPDOWN", "Line protocol on Interface GigabitEthernet1/0/{port}, changed state to {state}"),
    ("ILPOWER", 5, "DETECT", "Interface Gi1/0/{port}: Power Device detected: IEEE PD"),
    ("ILPOWER", 5, "IEEE_DISCONNECT", "Interface Gi1/0/{port}: PD removed"),
    ("ILPOWER", 5, "ILPOWER_POWER_DENY", "Interface Gi1/0/{port}: inline power denied. Reason: insufficient power"),
    ("ILPOWER", 5, "POWER_GRANTED", "Interface Gi1/0/{port}: Power granted"),
    ("SYS", 5, "CONFIG_I", "Configured from console by {user} on vty0 ({src})"),
    # access
    ("SEC_LOGIN", 5, "LOGIN_SUCCESS", "Login Success [user: {user}] [Source: {src}] [localport: {lport}] at {hhmmss} UTC {wday} {mon} {day} {year}"),
    ("SEC_LOGIN", 4, "LOGIN_FAILED", "Login failed [user: {user}] [Source: {src}] [localport: {lport}] [Reason: Login Authentication Failed] at {hhmmss} UTC {wday} {mon} {day} {year}"),
    ("SYS", 6, "LOGOUT", "User {user} has exited tty session 1({src})"),
    ("SYS", 6, "TTY_EXPIRE_TIMER", "(exec timer expired, tty 1 ({src})), user {user}"),
    ("SSH", 5, "SSH2_USERAUTH", "User '{user}' authentication for SSH2 Session from {src} (tty = 0) using crypto cipher 'aes256-ctr', hmac 'hmac-sha2-256'"),
    ("SSH", 4, "SSH2_UNEXPECTED_MSG", "Unexpected message type has arrived. Terminating the connection from {src}"),
    ("RADIUS", 4, "RADIUS_DEAD", "RADIUS server {aaa}:1812 is not responding."),
    ("RADIUS", 5, "RADIUS_LIVE", "RADIUS server {aaa}:1812 is responding again"),
    ("RADIUS", 3, "ALLDEADSERVER", "Group aaa-radius: No active radius servers found."),
    ("TACACS", 3, "TACACS_SERVER_DOWN", "TACACS+ server {aaa} is DOWN"),
    ("AAA", 5, "USER_LOCKED", "User {user} locked out on authentication failure from {src}"),
    # crypto / snmp
    ("SNMP", 3, "AUTHFAIL", "Authentication failure for SNMP req from host {src}"),
    ("SNMP", 5, "WARMSTART", "SNMP agent on host {host} is undergoing a warm start"),
    ("SNMP", 5, "COLDSTART", "SNMP agent on host {host} is undergoing a cold start"),
    ("CRYPTO", 4, "IKMP_NO_SA", "IKE SA was not found for peer {peer}"),
    ("CRYPTO", 5, "IKEV2_SESSION_STATUS", "IKEv2 session created with peer {peer}"),
    ("PKI", 3, "CERTIFICATE_INVALID", "Certificate validation failed for trustpoint TP-site"),
    ("SSH", 3, "KEY_SIZE_TOO_SMALL", "RSA key size 1024 bits is below recommended minimum for SSH"),
    # supplier / neighbor
    ("CDP", 4, "NATIVE_VLAN_MISMATCH", "Native VLAN mismatch discovered on GigabitEthernet1/0/{port} ({vlan_a}), with {nbr} GigabitEthernet1/0/{nport} ({vlan_b})."),
    ("CDP", 4, "DUPLEX_MISMATCH", "Duplex mismatch discovered on GigabitEthernet1/0/{port} (full), with {nbr} GigabitEthernet1/0/{nport} (half)."),
    ("LLDP", 5, "UPDATED", "Neighbor entry updated on GigabitEthernet1/0/{port}: Chassis-ID {mac}, Port-ID Gi1/0/{nport}"),
    ("PNP", 6, "PNP_PROFILE_CREATED", "PnP profile created for remote host device-helper.local"),
    ("SYS", 6, "LOGGINGHOST_STARTSTOP", "Logging to host {loghost} port 514 started - CLI initiated"),
    # patch
    ("PLATFORM", 5, "RESET_REASON", "Reset Reason: Soft Reset (Reload command)"),
    ("IOSXE", 5, "PLATFORM", "Switch 1 R0/0: pman: Package software integrity check completed"),
    ("INSTALL", 5, "INSTALL_COMPLETED_INFO", "Completed install add package flash:cat9k_iosxe.17.09.04a.SPA.bin"),
    ("BOOT", 5, "BOOTLOADER_UPGRADE", "Bootloader upgrade completed successfully"),
    # malware / port security
    ("PORT_SECURITY", 2, "PSECURE_VIOLATION", "Security violation occurred, caused by MAC address {macdot} on port GigabitEthernet1/0/{port}."),
    ("PM", 4, "ERR_DISABLE", "psecure-violation error detected on Gi1/0/{port}, putting Gi1/0/{port} in err-disable state"),
    ("IPS", 4, "SIGNATURE", "Sig:{sig} Subsig:0 Sev:{sev} TCP SYN Portscan [{attacker} -> {victim}]"),
    ("FW", 3, "MALWARE_BLOCK", "Threat signature matched; connection from {attacker} blocked on VLAN {vlan_a}"),
]

USERS = ["cisco", "netops", "ops-audit", "admin", "guest", "root", "auditor"]
SRCS = [
    "10.174.1.1", "10.174.0.177", "10.174.0.183", "10.174.0.195",
    "10.174.0.225", "10.174.100.100", "10.174.0.88", "10.174.0.90",
    "192.168.3.254", "10.174.100.103",
]
AAA = ["10.174.100.20", "10.174.100.21", "10.174.100.22"]
PEERS = ["10.174.100.50", "10.174.100.51", "10.174.100.60"]
NBRS = ["SW-CORE-01", "SW-ACCESS-02", "SW-OT-GW", "FW-EDGE-01"]
ATTACKERS = ["185.220.101.44", "203.0.113.45", "198.51.100.77", "45.33.32.156"]
LOGHOSTS = ["10.174.100.100", "10.174.100.101"]
WDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def count_events(path: Path) -> int:
    n = 0
    with path.open(encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if EVENT_RE.match(ln.strip()):
                n += 1
    return n


def fmt_ts(dt: datetime) -> str:
    # Cisco-like: "Jul 31 09:15:47 UTC"
    return f"{MONTHS[dt.month - 1]} {dt.day:2d} {dt.strftime('%H:%M:%S')} UTC"


def render_line(dt: datetime, host: str, victim_ip: str, rng: random.Random) -> str:
    fac, sev, mne, body_tpl = rng.choice(TEMPLATES)
    port = rng.randint(1, 48)
    nport = rng.randint(1, 48)
    user = rng.choice(USERS)
    src = rng.choice(SRCS)
    ctx = {
        "port": port,
        "nport": nport,
        "state": rng.choice(["up", "down"]),
        "user": user,
        "src": src,
        "lport": rng.choice([22, 23]),
        "aaa": rng.choice(AAA),
        "host": host,
        "peer": rng.choice(PEERS),
        "nbr": rng.choice(NBRS),
        "vlan_a": rng.choice([10, 20, 30, 40]),
        "vlan_b": rng.choice([10, 20, 30, 40]),
        "mac": ":".join(f"{rng.randint(0, 255):02x}" for _ in range(6)),
        "macdot": f"{rng.randint(0, 255):04x}.{rng.randint(0, 255):04x}.{rng.randint(0, 255):04x}",
        "loghost": rng.choice(LOGHOSTS),
        "attacker": rng.choice(ATTACKERS),
        "victim": victim_ip,
        "sig": rng.choice([2000, 2001, 2004, 3050]),
        "sev": rng.choice([20, 25, 30, 40]),
        "hhmmss": dt.strftime("%H:%M:%S"),
        "wday": WDAYS[dt.weekday()],
        "mon": MONTHS[dt.month - 1],
        "day": dt.day,
        "year": dt.year,
    }
    body = body_tpl.format(**ctx)
    return f"{fmt_ts(dt)}: %{fac}-{sev}-{mne}: {body}"


def expand_file(path: Path, need: int, host: str, victim_ip: str, seed: int) -> int:
    if need <= 0:
        return 0
    rng = random.Random(seed)
    start = datetime(2025, 6, 1, 0, 0, 0)
    # 分散在約一年內
    span_sec = int(timedelta(days=400).total_seconds())
    batch = []
    for i in range(need):
        dt = start + timedelta(seconds=rng.randint(0, span_sec))
        # 微調避免完全同一秒撞太多
        dt = dt + timedelta(seconds=(i % 17))
        batch.append(render_line(dt, host, victim_ip, rng))
    # 依時間排序再寫入，較像真實 buffer
    def sort_key(line: str):
        m = EVENT_RE.match(line)
        return m.group(1) if m else line

    batch.sort(key=sort_key)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write("\n")
        for ln in batch:
            f.write(ln + "\n")
    return need


def main():
    files = sorted(ROOT.glob("*.txt"))
    files = [p for p in files if p.name.lower() != "expand_logs_to_50k.py"]
    if not files:
        raise SystemExit(f"No .txt logs under {ROOT}")

    current = {p: count_events(p) for p in files}
    total = sum(current.values())
    print("Before:", current, "total=", total)
    if total >= TARGET_TOTAL:
        print(f"Already >= {TARGET_TOTAL}, skip.")
        return

    remain = TARGET_TOTAL - total
    # 平均分到各檔；餘數給第一個
    per = remain // len(files)
    extra = remain % len(files)

    meta = {
        "C9300-24p_192.168.3.3.2_flash.txt": ("C9300-24p", "192.168.3.3"),
        "C9300-48p_192.168.3.254_flash.txt": ("C9300-48p", "192.168.3.254"),
    }

    for i, p in enumerate(files):
        need = per + (extra if i == 0 else 0)
        host, vip = meta.get(p.name, (p.stem.split("_")[0], "192.168.3.1"))
        added = expand_file(p, need, host, vip, seed=1000 + i)
        print(f"Appended {added} events -> {p.name}")

    after = {p.name: count_events(p) for p in files}
    print("After:", after, "total=", sum(after.values()))


if __name__ == "__main__":
    main()
