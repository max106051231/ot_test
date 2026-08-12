"""
擴充 OT Cisco flash 日誌：更多元事件 + 更高覆蓋（半導體廠場域）。

用法（專案根目錄）：
  python ot/expand_logs_diverse.py
  python ot/expand_logs_diverse.py --target 100000
"""
from __future__ import annotations

import argparse
import random
import re
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVENT_RE = re.compile(
    r"^([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}(?:\s+[A-Z]+)?)\s*:\s*"
    r"%([A-Z0-9_]+)-(\d+)-([A-Z0-9_]+):\s*(.*)$"
)

# (weight, facility, severity, mnemonic, body_template, control_hint)
# control_hint 僅供統計，實際分類仍靠 app.py 規則
TEMPLATES = [
    # ---- access_control ----
    (8, "SEC_LOGIN", 5, "LOGIN_SUCCESS",
     "Login Success [user: {user}] [Source: {src}] [localport: {lport}] at {hhmmss} UTC {wday} {mon} {day} {year}"),
    (10, "SEC_LOGIN", 4, "LOGIN_FAILED",
     "Login failed [user: {user}] [Source: {src}] [localport: {lport}] [Reason: Login Authentication Failed] at {hhmmss} UTC {wday} {mon} {day} {year}"),
    (4, "SEC_LOGIN", 3, "LOGIN_FAILED",
     "Login failed [user: {user}] [Source: {src}] [localport: 22] [Reason: Account locked] at {hhmmss} UTC {wday} {mon} {day} {year}"),
    (5, "SYS", 6, "LOGOUT", "User {user} has exited tty session 1({src})"),
    (3, "SYS", 6, "TTY_EXPIRE_TIMER", "(exec timer expired, tty 1 ({src})), user {user}"),
    (6, "SSH", 5, "SSH2_USERAUTH",
     "User '{user}' authentication for SSH2 Session from {src} (tty = 0) using crypto cipher 'aes256-ctr', hmac 'hmac-sha2-256'"),
    (4, "SSH", 4, "SSH2_UNEXPECTED_MSG",
     "Unexpected message type has arrived. Terminating the connection from {src}"),
    (3, "SSH", 3, "SSH2_ALGO_MISMATCH",
     "SSH2 algorithm negotiation failed with {src}; peer proposed weak ciphers"),
    (5, "RADIUS", 4, "RADIUS_DEAD", "RADIUS server {aaa}:1812 is not responding."),
    (3, "RADIUS", 5, "RADIUS_LIVE", "RADIUS server {aaa}:1812 is responding again"),
    (3, "RADIUS", 3, "ALLDEADSERVER", "Group aaa-radius: No active radius servers found."),
    (3, "RADIUS", 4, "RADIUS_AUTH_FAIL",
     "RADIUS authentication failed for user {user} from {src} (NAS={host})"),
    (3, "TACACS", 3, "TACACS_SERVER_DOWN", "TACACS+ server {aaa} is DOWN"),
    (2, "TACACS", 4, "TACACS_AUTHEN_FAIL",
     "TACACS+ authentication FAIL for user {user} from {src}"),
    (4, "AAA", 5, "USER_LOCKED", "User {user} locked out on authentication failure from {src}"),
    (3, "AAA", 4, "AUTHEN_ERROR",
     "AAA authentication error for user {user} from {src}: server unreachable"),
    (2, "DOT1X", 4, "AUTH_FAIL",
     "Authentication failed for client {macdot} on Interface GigabitEthernet1/0/{port}"),
    (3, "DOT1X", 5, "AUTH_SUCCESS",
     "Authentication successful for client {macdot} on Interface GigabitEthernet1/0/{port} VLAN {vlan_a}"),
    (2, "MAB", 4, "MAB_FAIL",
     "MAC Authentication Bypass failed for {macdot} on Gi1/0/{port}"),
    (2, "LOCAL", 4, "LOCAL_LOGIN_FAIL",
     "Local login authentication failed for user {user} from console"),

    # ---- sec_gem_log (crypto / snmp / tls) ----
    (6, "SNMP", 3, "AUTHFAIL", "Authentication failure for SNMP req from host {src}"),
    (3, "SNMP", 5, "WARMSTART", "SNMP agent on host {host} is undergoing a warm start"),
    (2, "SNMP", 5, "COLDSTART", "SNMP agent on host {host} is undergoing a cold start"),
    (3, "SNMP", 4, "COMMUNITY_MISMATCH",
     "SNMP community string mismatch from {src} (expected RO-semicon)"),
    (4, "CRYPTO", 4, "IKMP_NO_SA", "IKE SA was not found for peer {peer}"),
    (3, "CRYPTO", 5, "IKEV2_SESSION_STATUS", "IKEv2 session created with peer {peer}"),
    (3, "CRYPTO", 3, "IKEV2_AUTH_FAIL",
     "IKEv2 authentication failed with peer {peer}; proposal rejected"),
    (2, "CRYPTO", 4, "IPSEC_SA_EXPIRED",
     "IPSec SA expired for peer {peer}; renegotiation started"),
    (3, "PKI", 3, "CERTIFICATE_INVALID", "Certificate validation failed for trustpoint TP-site"),
    (2, "PKI", 4, "CERT_EXPIRING",
     "Certificate for trustpoint TP-fab will expire in {days} days"),
    (2, "PKI", 3, "CRL_FAILURE",
     "Unable to retrieve CRL for trustpoint TP-fab from {peer}"),
    (2, "SSH", 3, "KEY_SIZE_TOO_SMALL",
     "RSA key size 1024 bits is below recommended minimum for SSH"),
    (2, "TLS", 4, "TLS_HANDSHAKE_FAIL",
     "TLS handshake failed with {src} (cipher suite not allowed)"),
    (2, "SSL", 3, "SSL_CERT_VERIFY_FAIL",
     "SSL certificate verification failed for peer {peer}"),
    (2, "SUDI", 5, "SUDI_VALID",
     "Secure Unique Device Identifier validated for switch {host}"),

    # ---- patch_management ----
    (4, "PLATFORM", 5, "RESET_REASON", "Reset Reason: Soft Reset (Reload command)"),
    (3, "PLATFORM", 4, "RESET_REASON", "Reset Reason: Power Failure / Unexpected reload"),
    (4, "IOSXE", 5, "PLATFORM", "Switch 1 R0/0: pman: Package software integrity check completed"),
    (3, "IOSXE", 3, "PLATFORM",
     "Switch 1 R0/0: pman: Software integrity check WARNING - signature mismatch on package"),
    (4, "INSTALL", 5, "INSTALL_COMPLETED_INFO",
     "Completed install add package flash:cat9k_iosxe.{ver}.SPA.bin"),
    (3, "INSTALL", 4, "INSTALL_START",
     "Starting install activate package flash:cat9k_iosxe.{ver}.SPA.bin"),
    (2, "INSTALL", 3, "INSTALL_ABORT",
     "Install aborted: insufficient free space on flash: for {ver}"),
    (3, "BOOT", 5, "BOOTLOADER_UPGRADE", "Bootloader upgrade completed successfully"),
    (2, "BOOT", 4, "BOOT_IMAGE",
     "Booting system image flash:cat9k_iosxe.{ver}.SPA.bin"),
    (2, "VERSION", 6, "SYSINFO",
     "Cisco IOS XE Software, Version {ver} on {host}"),
    (2, "SOFTWARE", 4, "SMU_INSTALL",
     "SMU patch {smu} installed successfully on {host}"),
    (2, "UPGRADE", 3, "FW_MISMATCH",
     "Field-programmable device firmware mismatch detected on module 1"),
    (2, "IMAGE", 4, "IMAGE_CHECKSUM",
     "Image checksum verification OK for cat9k_iosxe.{ver}.SPA.bin"),

    # ---- supplier_security ----
    (4, "CDP", 4, "NATIVE_VLAN_MISMATCH",
     "Native VLAN mismatch discovered on GigabitEthernet1/0/{port} ({vlan_a}), with {nbr} GigabitEthernet1/0/{nport} ({vlan_b})."),
    (3, "CDP", 4, "DUPLEX_MISMATCH",
     "Duplex mismatch discovered on GigabitEthernet1/0/{port} (full), with {nbr} GigabitEthernet1/0/{nport} (half)."),
    (2, "CDP", 5, "DEVICE_DETECTED",
     "CDP neighbor {nbr} detected on GigabitEthernet1/0/{port}"),
    (4, "LLDP", 5, "UPDATED",
     "Neighbor entry updated on GigabitEthernet1/0/{port}: Chassis-ID {mac}, Port-ID Gi1/0/{nport}"),
    (2, "LLDP", 4, "REMOVED",
     "Neighbor {nbr} aged out on GigabitEthernet1/0/{port}"),
    (3, "PNP", 6, "PNP_PROFILE_CREATED",
     "PnP profile created for remote host device-helper.local"),
    (3, "SYS", 6, "LOGGINGHOST_STARTSTOP",
     "Logging to host {loghost} port 514 started - CLI initiated"),
    (2, "SYS", 5, "LOGGINGHOST_STARTSTOP",
     "Logging to host {loghost} port 514 stopped - connectivity lost"),
    (2, "TRAP", 4, "SNMP_TRAP_SEND_FAIL",
     "Failed to send SNMP trap to remote-host {loghost}"),
    (2, "NEIGHBOR", 4, "BGP_ADJCHANGE",
     "Neighbor {peer} Down - Peer closed the session (supplier VPN)"),
    (2, "REMOTE", 5, "REMOTE_HOST_ACCESS",
     "Remote host {src} accessed management plane via jump-host {peer}"),

    # ---- malware_defense ----
    (5, "PORT_SECURITY", 2, "PSECURE_VIOLATION",
     "Security violation occurred, caused by MAC address {macdot} on port GigabitEthernet1/0/{port}."),
    (4, "PM", 4, "ERR_DISABLE",
     "psecure-violation error detected on Gi1/0/{port}, putting Gi1/0/{port} in err-disable state"),
    (3, "PM", 4, "ERR_DISABLE",
     "bpduguard error detected on Gi1/0/{port}, putting Gi1/0/{port} in err-disable state"),
    (5, "IPS", 4, "SIGNATURE",
     "Sig:{sig} Subsig:0 Sev:{sev} TCP SYN Portscan [{attacker} -> {victim}]"),
    (3, "IPS", 3, "SIGNATURE",
     "Sig:{sig} Subsig:0 Sev:{sev} SMB lateral movement attempt [{attacker} -> {victim}]"),
    (3, "IPS", 2, "SIGNATURE",
     "Sig:{sig} Subsig:0 Sev:{sev} Exploit kit HTTP payload [{attacker} -> {victim}]"),
    (4, "FW", 3, "MALWARE_BLOCK",
     "Threat signature matched; connection from {attacker} blocked on VLAN {vlan_a}"),
    (3, "FW", 3, "MALWARE_BLOCK",
     "Malware C2 domain blocked; source {src} destination {attacker} on VLAN {vlan_a}"),
    (2, "IDS", 4, "THREAT",
     "Anomalous DNS tunneling suspected from {src} to {attacker}"),
    (2, "DOS", 3, "HOST_ATTACK",
     "Possible DoS flood detected from {attacker} toward {victim} rate {rate} pps"),
    (2, "DDOS", 2, "ATTACK_DETECTED",
     "Distributed scan pattern from {attacker} and peers toward fab management VLAN {vlan_a}"),
    (2, "USB", 3, "USB_DEVICE",
     "Unauthorized USB storage device detected on switch {host} (blocked by policy)"),
    (2, "VIRUS", 3, "AV_ALERT",
     "Endpoint AV reported threat on host {src}; quarantine recommended"),

    # ---- recipe_audit (config / link / poe / spanning) ----
    (8, "LINK", 3, "UPDOWN",
     "Interface GigabitEthernet1/0/{port}, changed state to {state}"),
    (6, "LINEPROTO", 5, "UPDOWN",
     "Line protocol on Interface GigabitEthernet1/0/{port}, changed state to {state}"),
    (4, "ILPOWER", 5, "DETECT", "Interface Gi1/0/{port}: Power Device detected: IEEE PD"),
    (3, "ILPOWER", 5, "IEEE_DISCONNECT", "Interface Gi1/0/{port}: PD removed"),
    (4, "ILPOWER", 5, "ILPOWER_POWER_DENY",
     "Interface Gi1/0/{port}: inline power denied. Reason: insufficient power"),
    (3, "ILPOWER", 5, "POWER_GRANTED", "Interface Gi1/0/{port}: Power granted"),
    (5, "SYS", 5, "CONFIG_I",
     "Configured from console by {user} on vty0 ({src})"),
    (4, "SYS", 5, "CONFIG_I",
     "Configured from console by {user} on vty2 ({src})"),
    (3, "PARSER", 5, "CMD",
     "CMD: '{cmd}' by {user} from {src}"),
    (2, "PARSER", 4, "CMD_DENIED",
     "CMD denied: '{cmd}' by {user} from {src} (privilege insufficient)"),
    (3, "SPANTREE", 4, "ROOTCHANGE",
     "VLAN {vlan_a} root changed to {macdot} on GigabitEthernet1/0/{port}"),
    (2, "SPANTREE", 3, "BLOCK",
     "VLAN {vlan_a} GigabitEthernet1/0/{port} -> blocking"),
    (2, "DUAL_ACTIVE", 3, "DETECTION",
     "Dual-active detection on stack member; GigabitEthernet1/0/{port} errdisabled"),
    (3, "PORT_SECURITY", 4, "PSECURE_VIOLATION",
     "Address {macdot} aged out / max MAC exceeded on GigabitEthernet1/0/{port}"),
    (2, "SYS", 4, "CONFIG_NVGEN",
     "Configuration write memory completed by {user}"),
    (2, "SYS", 5, "RELOAD",
     "Reload requested by {user} from {src} reason: scheduled maintenance window"),
    # semiconductor fab flavored operational events
    (3, "LINK", 3, "UPDOWN",
     "Interface TenGigabitEthernet1/1/{port}, changed state to {state} (tool AMHS-{tool})"),
    (2, "SYS", 5, "CONFIG_I",
     "Configured from console by {user} on vty0 ({src}) ; change-id={changeid}"),
    (2, "ILPOWER", 4, "POWER_BUDGET",
     "PoE power budget warning on switch {host}: used {poe_used}W / {poe_total}W"),
    (2, "ENV", 4, "FAN_FAIL",
     "Fan {fan} failure detected on switch {host} (cleanroom rack {rack})"),
    (2, "ENV", 3, "TEMP_ALERT",
     "Temperature alert on switch {host}: inlet {temp}C exceeds threshold"),
]

USERS = [
    "cisco", "netops", "ops-audit", "admin", "guest", "auditor",
    "fab-netops", "efem-tech", "cim-admin", "sec-soc", "vendor-se",
    "shift-a", "shift-b", "maint01", "tooleng",
]
SRCS = [
    "10.174.1.1", "10.174.0.177", "10.174.0.183", "10.174.0.195",
    "10.174.0.225", "10.174.100.100", "10.174.0.88", "10.174.0.90",
    "192.168.3.254", "10.174.100.103", "10.50.20.11", "10.50.20.45",
    "10.60.1.8", "10.60.1.22", "172.20.15.40",
]
AAA = ["10.174.100.20", "10.174.100.21", "10.174.100.22", "10.50.5.10"]
PEERS = ["10.174.100.50", "10.174.100.51", "10.174.100.60", "10.50.8.2"]
NBRS = [
    "SW-CORE-01", "SW-ACCESS-02", "SW-OT-GW", "FW-EDGE-01",
    "SW-FAB-AMHS", "SW-CLEAN-L2", "SW-TOOL-BAY3",
]
ATTACKERS = [
    "185.220.101.44", "203.0.113.45", "198.51.100.77", "45.33.32.156",
    "91.219.237.12", "103.27.188.55",
]
LOGHOSTS = ["10.174.100.100", "10.174.100.101", "10.50.9.20"]
CMDS = [
    "show running-config", "write memory", "configure terminal",
    "interface GigabitEthernet1/0/12", "no shutdown",
    "aaa new-model", "snmp-server community public RO",
    "reload", "clear port-security all",
]
VERS = ["17.09.04a", "17.09.05", "17.12.01", "17.06.05"]
SMUS = ["cat9k_iosxe.17.09.04a.CSCxx12345.SPA.smu.bin", "cat9k_iosxe.17.12.01.CSCyy999.SPA.smu.bin"]
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
    return f"{MONTHS[dt.month - 1]} {dt.day:2d} {dt.strftime('%H:%M:%S')} UTC"


def _weighted_choice(rng: random.Random):
    weights = [t[0] for t in TEMPLATES]
    return rng.choices(TEMPLATES, weights=weights, k=1)[0]


def render_line(dt: datetime, host: str, victim_ip: str, rng: random.Random) -> str:
    _w, fac, sev, mne, body_tpl = _weighted_choice(rng)
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
        "lport": rng.choice([22, 23, 443]),
        "aaa": rng.choice(AAA),
        "host": host,
        "peer": rng.choice(PEERS),
        "nbr": rng.choice(NBRS),
        "vlan_a": rng.choice([10, 20, 30, 40, 100, 110, 200]),
        "vlan_b": rng.choice([10, 20, 30, 40, 100, 110]),
        "mac": ":".join(f"{rng.randint(0, 255):02x}" for _ in range(6)),
        "macdot": f"{rng.randint(0, 255):04x}.{rng.randint(0, 255):04x}.{rng.randint(0, 255):04x}",
        "loghost": rng.choice(LOGHOSTS),
        "attacker": rng.choice(ATTACKERS),
        "victim": victim_ip,
        "sig": rng.choice([2000, 2001, 2004, 2150, 3050, 6050, 7100]),
        "sev": rng.choice([20, 25, 30, 40, 50]),
        "hhmmss": dt.strftime("%H:%M:%S"),
        "wday": WDAYS[dt.weekday()],
        "mon": MONTHS[dt.month - 1],
        "day": dt.day,
        "year": dt.year,
        "cmd": rng.choice(CMDS),
        "ver": rng.choice(VERS),
        "smu": rng.choice(SMUS),
        "days": rng.randint(7, 90),
        "rate": rng.randint(5000, 80000),
        "tool": rng.randint(101, 399),
        "changeid": f"CHG{rng.randint(10000, 99999)}",
        "poe_used": rng.randint(600, 1100),
        "poe_total": 1200,
        "fan": rng.randint(1, 4),
        "rack": f"R{rng.randint(1, 24):02d}",
        "temp": rng.randint(48, 72),
    }
    body = body_tpl.format(**ctx)
    return f"{fmt_ts(dt)}: %{fac}-{sev}-{mne}: {body}"


def expand_file(path: Path, need: int, host: str, victim_ip: str, seed: int) -> int:
    if need <= 0:
        return 0
    rng = random.Random(seed)
    start = datetime(2025, 1, 1, 0, 0, 0)
    span_sec = int(timedelta(days=500).total_seconds())
    batch = []
    for i in range(need):
        dt = start + timedelta(seconds=rng.randint(0, span_sec) + (i % 23))
        batch.append(render_line(dt, host, victim_ip, rng))

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=100_000, help="目標總事件數（預設 100000）")
    ap.add_argument("--add", type=int, default=0, help="不指定總量時，各檔合計再追加 N 筆")
    args = ap.parse_args()

    files = sorted(p for p in ROOT.glob("*.txt") if p.is_file())
    if not files:
        raise SystemExit(f"No .txt logs under {ROOT}")

    current = {p: count_events(p) for p in files}
    total = sum(current.values())
    print("Before:", {p.name: n for p, n in current.items()}, "total=", total)
    print(f"Template varieties: {len(TEMPLATES)}")

    if args.add > 0:
        remain = args.add
    else:
        if total >= args.target:
            print(f"Already >= {args.target}. Use --add N to force append more.")
            return
        remain = args.target - total

    per = remain // len(files)
    extra = remain % len(files)
    meta = {
        "C9300-24p_192.168.3.3.2_flash.txt": ("C9300-24p", "192.168.3.3"),
        "C9300-48p_192.168.3.254_flash.txt": ("C9300-48p", "192.168.3.254"),
    }

    for i, p in enumerate(files):
        need = per + (extra if i == 0 else 0)
        host, vip = meta.get(p.name, (p.stem.split("_")[0], "192.168.3.1"))
        added = expand_file(p, need, host, vip, seed=20260803 + i)
        print(f"Appended {added} diverse events -> {p.name}")

    after = {p.name: count_events(p) for p in files}
    print("After:", after, "total=", sum(after.values()))


if __name__ == "__main__":
    main()
