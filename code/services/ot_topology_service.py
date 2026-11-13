"""OT 場域拓樸：多廠區、Port 連線表、日誌設備狀態。"""
from __future__ import annotations

import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

from code.paths import config_dir, data_dir

_TOPO_CONFIG = config_dir() / "ot_topology_sites.json"
_LAYOUT_SCHEMA = "ot-topology-layout-v1"


def _layout_dir() -> Path:
    d = data_dir() / "ot_topology_layouts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _layout_file(site_id: str) -> Path:
    sid = re.sub(r"[^a-zA-Z0-9_-]", "_", (site_id or "fab-a").strip()) or "fab-a"
    return _layout_dir() / f"{sid}.json"


def apply_layout_positions(nodes: list[dict], layout_nodes: list[dict]) -> int:
    """將自訂 x,y 套到節點；回傳套用數。"""
    if not nodes or not layout_nodes:
        return 0
    pos = {
        str(item.get("id") or ""): item
        for item in layout_nodes
        if isinstance(item, dict) and item.get("id")
    }
    applied = 0
    for n in nodes:
        nid = n.get("id")
        p = pos.get(nid)
        if not p:
            continue
        try:
            x, y = float(p.get("x")), float(p.get("y"))
        except (TypeError, ValueError):
            continue
        n["x"] = x
        n["y"] = y
        applied += 1
    return applied


def load_custom_layout(site_id: str) -> dict | None:
    path = _layout_file(site_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def save_custom_layout(
    site_id: str,
    nodes: list[dict],
    *,
    label: str = "",
    updated_by: str = "",
) -> dict:
    sid = (site_id or "fab-a").strip()
    cleaned: list[dict] = []
    for item in nodes or []:
        if not isinstance(item, dict):
            continue
        nid = str(item.get("id") or "").strip()
        if not nid:
            continue
        try:
            x, y = float(item.get("x")), float(item.get("y"))
        except (TypeError, ValueError):
            continue
        cleaned.append({"id": nid, "x": round(x, 2), "y": round(y, 2)})

    payload = {
        "schema": _LAYOUT_SCHEMA,
        "site_id": sid,
        "nodes": cleaned,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label": (label or "").strip()[:120],
        "updated_by": (updated_by or "").strip()[:80],
    }
    path = _layout_file(sid)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def delete_custom_layout(site_id: str) -> bool:
    path = _layout_file(site_id)
    if not path.is_file():
        return False
    path.unlink()
    return True


def _load_topo_config() -> dict:
    if not _TOPO_CONFIG.is_file():
        return {"sites": [], "layers": []}
    return json.loads(_TOPO_CONFIG.read_text(encoding="utf-8"))


def list_topology_sites() -> list[dict]:
    cfg = _load_topo_config()
    return [{"id": s["id"], "name": s.get("name") or s["id"]} for s in cfg.get("sites") or []]


def _site_config(site_id: str) -> dict | None:
    sid = (site_id or "fab-a").strip().lower()
    for s in _load_topo_config().get("sites") or []:
        if str(s.get("id", "")).lower() == sid:
            return s
    return None


_ANNEX_TAGS = (
    "A.8.22", "A.8.20", "A.8.19", "A.8.16", "A.8.15", "A.8.9", "A.8.5",
    "A.5.15", "A.5.14", "A.5.10", "A.5.28", "A.5.33",
)

_SYSLOG_BY_TYPE: dict[str, list[str]] = {
    "switch": [
        "%LINK-3-UPDOWN: Interface GigabitEthernet1/0/{port}, changed state to {state}",
        "%LINEPROTO-5-UPDOWN: Line protocol on Interface Gi1/0/{port}, changed state to up",
        "%SEC-6-IPACCESSLOGP: list OT-ACL {action} {proto} {src} -> {dst}, {count} packets",
        "%SPAN-6-ENABLE: Spanning tree forwarding on Gi1/0/{port}, VLAN {vlan}",
        "%PM-4-ERR_DISABLE: err-disable on Gi1/0/{port}, reason: {reason}",
    ],
    "firewall": [
        "ASDM-6-302013: Built {dir} {proto} connection {src}:{sport} -> {dst}:{dport}",
        "ASA-6-302021: Teardown {proto} connection {src}:{sport} / {dst}:{dport} duration {dur}s",
        "%FTD-6-430005: Deny IP due to ACL policy; src {src} dst {dst}",
    ],
    "server": [
        "sshd[{pid}]: Accepted password for otadmin from {src} port {sport} ssh2",
        "systemd[1]: Started Semi-Shield log forwarder to collector",
        "audit[{pid}]: USER_LOGIN pid={pid} uid=0 auid=1000 msg='op=PAM:session_open'",
    ],
    "scada": [
        "OPCUA-Server: Session created client={src} cert=HMI-Line subscription=1200ms",
        "Historian: Tag batch commit count={count} lag_ms={lag}",
        "AlarmMgr: {level} alarm tag=Cell.{cell}.Temp value={val} limit={lim}",
    ],
    "hmi": [
        "HMI-Gateway: Operator login user={user} station=Line-A shift=2",
        "HMI-Gateway: Recipe download ack PLC={plc} recipe={recipe} checksum=0x{cs}",
    ],
    "workstation": [
        "WinEvt: Config change detected application=Studio5000 user={user} host={host}",
        "EDR: Process blocked name=unknown.exe hash=SHA256:{hash}",
    ],
    "plc": [
        "Profinet-IO: Device {label} IP {ip} status=RUN cycle_ms={cycle}",
        "Profinet-IO: Module fault slot=3 code=0x{code} device={label}",
        "SafetyPLC: E-stop circuit OK channel=A/B device={label}",
    ],
    "robot": [
        "Robot-Cell: Cycle complete job={job} takt={takt}s faults=0 axis=6",
        "Robot-Cell: Minor fault code={code} reset by operator at {ip}",
    ],
    "gateway": [
        "IO-Gateway: Analog scan batch={count} devices online={online}/{total}",
        "IO-Gateway: MQTT publish topic=ot/cell/{cell}/telemetry qos=1 bytes={bytes}",
        "Vision-AOI: Inspection pass rate={rate}% lot={lot}",
    ],
    "siem": [
        "SIEM: Ingest eps={eps} dropped=0 source=OT-collector",
        "SIEM: Correlation rule OT-Anomaly matched src={src} dst={dst} severity={sev}",
    ],
    "collector": [
        "Collector: Received {eps} eps from {devices} devices queue_depth={qd}",
        "Collector: Normalized Cisco syslog file={file} lines={lines}",
    ],
}


def _default_telemetry(node_type: str, label: str, host: int | None) -> dict:
    base = {
        "switch": {"syslog_events": 1840, "ports_up": 24, "cpu_pct": 17, "mem_pct": 42},
        "firewall": {"sessions": 920, "deny_h": 3, "syslog_events": 640},
        "server": {"cpu_pct": 28, "mem_pct": 55, "disk_pct": 61},
        "scada": {"tags_live": 1840, "comm_ok_pct": 99.2, "alarms_active": 1},
        "hmi": {"sessions": 2, "recipe_ver": "R-2026.03.12"},
        "workstation": {"config_drift": 12, "patch_pending": 1},
        "plc": {"cycle_ms": 12 + (hash(label) % 9), "io_ok": 1, "tags": 420 + (host or 0) % 200},
        "robot": {"takt_s": 18, "faults_shifts": 0, "util_pct": 87},
        "gateway": {"devices_online": 16, "scan_ms": 45, "mqtt_eps": 120},
        "siem": {"eps": 2400, "alerts_h": 6},
        "collector": {"ingest_eps": 380, "files_buffered": 2},
    }
    return dict(base.get(node_type, {"syslog_events": 100, "status_ok": 1}))


class _SafeFormatDict(dict):
    """str.format 缺 key 時回退，避免模板欄位漏填導致 KeyError。"""

    def __missing__(self, key: str) -> str:
        return "?"


def _syslog_template_vars(rng: random.Random, *, ip: str, label: str, site_id: str, host_octet: str) -> dict:
    return {
        "port": rng.randint(1, 48),
        "state": rng.choice(["up", "down", "up"]),
        "action": rng.choice(["permit", "deny"]),
        "proto": rng.choice(["tcp", "udp", "icmp", "Profinet"]),
        "src": f"192.168.3.{rng.randint(100, 220)}",
        "dst": ip,
        "sport": rng.randint(1024, 65000),
        "dport": rng.choice([22, 80, 443, 4840, 44818, 502]),
        "count": rng.randint(1, 240),
        "vlan": rng.choice([110, 120, 121, 99, 100]),
        "reason": rng.choice(["bpdu-guard", "link-flap", "psecure-violation"]),
        "dir": rng.choice(["inbound", "outbound"]),
        "dur": rng.randint(4, 3600),
        "pid": rng.randint(1000, 32000),
        "user": rng.choice(["otadmin", "eng_ro", "shift2"]),
        "host": host_octet,
        "level": rng.choice(["WARN", "INFO", "CRIT"]),
        "cell": rng.randint(1, 8),
        "val": round(rng.uniform(42.0, 88.0), 1),
        "lim": 85.0,
        "lag": rng.randint(80, 480),
        "plc": label[:12],
        "recipe": rng.choice(["PKG-A01", "LITHO-B2", "ETCH-C3"]),
        "cs": rng.randint(0x1000, 0xFFFF),
        "hash": rng.randint(0x10_000_000, 0xFF_FFFF_FFFF),
        "job": rng.choice(["pick-place", "wirebond", "vision-scan"]),
        "takt": rng.randint(14, 28),
        "code": rng.randint(1, 99),
        "online": rng.randint(12, 24),
        "total": 24,
        "bytes": rng.randint(128, 4096),
        "rate": round(rng.uniform(96.5, 99.9), 2),
        "lot": rng.choice(["LOT-8842", "LOT-9011", "LOT-7720"]),
        "eps": rng.randint(120, 4200),
        "sev": rng.choice([2, 3, 4, 5, 6]),
        "devices": rng.randint(8, 32),
        "qd": rng.randint(0, 12),
        "file": f"sim_{site_id}_{rng.randint(1, 9)}.txt",
        "lines": rng.randint(200, 8000),
        "cycle": rng.randint(8, 22),
        "label": label,
        "ip": ip,
    }


def generate_virtual_syslog_feed(
    nodes: list[dict],
    *,
    site_id: str = "fab-a",
    count: int = 48,
) -> list[dict]:
    """依拓樸節點產生虛擬 Cisco / OT syslog，模擬完整場域回報。"""
    if not nodes:
        return []
    rng = random.Random(f"{site_id}:{int(time.time()) // 120}")
    out: list[dict] = []
    for _ in range(count):
        node = rng.choice(nodes)
        ntype = node.get("type") or "server"
        templates = _SYSLOG_BY_TYPE.get(ntype) or _SYSLOG_BY_TYPE["server"]
        msg_tpl = rng.choice(templates)
        ip = node.get("ip") or "192.168.0.1"
        label = node.get("label") or node.get("id") or "device"
        parts = ip.split(".")
        host_octet = parts[-1] if parts else "10"
        fmt = _SafeFormatDict(_syslog_template_vars(rng, ip=ip, label=label, site_id=site_id, host_octet=host_octet))
        msg = msg_tpl.format_map(fmt)
        sev = 5
        if "%ERR_DISABLE" in msg or "Deny" in msg or "fault" in msg.lower():
            sev = rng.choice([2, 3, 4])
        elif "WARN" in msg or "Minor" in msg:
            sev = 4
        out.append({
            "time": time.strftime("%b %d %H:%M:%S UTC"),
            "severity": sev,
            "device": label,
            "ip": ip,
            "summary": msg[:200],
            "control": rng.choice(_ANNEX_TAGS),
            "simulated": True,
        })
    return out


def _log_health(parsed: list, device: str, ip: str) -> str:
    needle = (device or "").lower()
    ip_s = ip or ""
    sev_min = 6
    hit = 0
    for ev in parsed:
        raw = (ev.get("raw") or "") + " " + (ev.get("message") or "")
        if ip_s and ip_s in raw:
            hit += 1
            sev_min = min(sev_min, int(ev.get("severity") or 5))
        elif needle and needle.lower() in raw.lower():
            hit += 1
            sev_min = min(sev_min, int(ev.get("severity") or 5))
    if hit == 0:
        return "ok"
    for ev in parsed:
        raw = (ev.get("raw") or "") + " " + (ev.get("message") or "")
        if ip_s and ip_s not in raw and needle and needle.lower() not in raw.lower():
            continue
        if re.search(r"ERR_DISABLE|ILPOWER|DENIED|FAILED", raw, re.I):
            return "warn" if sev_min > 2 else "fail"
    if sev_min <= 2:
        return "fail"
    if sev_min <= 3:
        return "warn"
    return "ok"


def sim_syslog_for_site(site_id: str, count: int = 8) -> list[dict]:
    """輕量虛擬 syslog 批次（即時串流用，不掃 ot/）。"""
    site_cfg = _site_config(site_id)
    if not site_cfg:
        return []
    subnet = site_cfg.get("subnet") or "192.168.3"
    nodes = [
        {
            "id": tpl["id"],
            "label": tpl.get("label") or tpl["id"],
            "type": tpl.get("type") or "server",
            "ip": f"{subnet}.{tpl.get('host', 1)}",
        }
        for tpl in (site_cfg.get("nodes") or [])
    ]
    return generate_virtual_syslog_feed(nodes, site_id=site_cfg["id"], count=max(1, min(count, 24)))


def _layer_centers(layers: list) -> dict[str, float]:
    out: dict[str, float] = {}
    for L in layers or []:
        lid = L.get("id")
        if not lid:
            continue
        out[lid] = (float(L.get("y0", 0)) + float(L.get("y1", 0))) / 2.0
    return out


def _layer_bounds(layers: list, layer_id: str) -> tuple[float, float]:
    for L in layers or []:
        if L.get("id") == layer_id:
            return float(L.get("y0", 0)), float(L.get("y1", 0))
    return 0.0, 100.0


def _auto_layout_site(site_cfg: dict, layers: list) -> dict[str, tuple[float, float]]:
    """
    Purdue 分層排版：加寬畫布、L0 依交換器順序拉開槽位，雙列錯開標籤區。
    """
    templates = list(site_cfg.get("nodes") or [])
    if not templates:
        return {}
    by_id = {n["id"]: n for n in templates}
    layer_y = _layer_centers(layers)
    positions: dict[str, tuple[float, float]] = {}

    canvas_w = 1220.0
    margin = 90.0

    def spread_x(count: int, index: int) -> float:
        if count <= 0:
            return canvas_w / 2
        usable = canvas_w - 2 * margin
        return margin + usable * (index + 0.5) / count

    for layer_id in ("L5", "L35", "L3", "L2"):
        ids = sorted(
            [n["id"] for n in templates if n.get("layer") == layer_id],
            key=lambda i: float(by_id[i].get("x") or 0),
        )
        y = layer_y.get(layer_id, 200.0)
        for i, nid in enumerate(ids):
            positions[nid] = (spread_x(len(ids), i), y)

    uplink: dict[str, str] = {}
    for link in site_cfg.get("port_links") or []:
        fr, to = link.get("from"), link.get("to")
        if fr not in by_id or to not in by_id:
            continue
        fl, tl = by_id[fr].get("layer"), by_id[to].get("layer")
        if fl == "L1" and tl == "L0":
            uplink[to] = fr
        elif tl == "L1" and fl == "L0":
            uplink[fr] = to

    l1_ids = [n["id"] for n in templates if n.get("layer") == "L1"]

    def l1_rank(nid: str) -> tuple:
        n = by_id[nid]
        if n.get("type") == "gateway":
            return (0, float(n.get("x") or 0))
        return (1, float(n.get("x") or 0))

    l1_ids.sort(key=l1_rank)
    y_l1 = layer_y.get("L1", 438.0)
    for i, nid in enumerate(l1_ids):
        positions[nid] = (spread_x(len(l1_ids), i), y_l1)

    groups: dict[str, list[str]] = defaultdict(list)
    orphans: list[str] = []
    for n in templates:
        if n.get("layer") != "L0":
            continue
        nid = n["id"]
        parent = uplink.get(nid)
        if parent and parent in by_id:
            groups[parent].append(nid)
        else:
            orphans.append(nid)

    l0_y0, l0_y1 = _layer_bounds(layers, "L0")
    y_l0_a = l0_y0 + 52.0
    y_l0_b = l0_y1 - 48.0

    ordered_l0: list[str] = []
    for pid in l1_ids:
        kids = sorted(groups.get(pid, []), key=lambda c: by_id[c].get("label") or c)
        ordered_l0.extend(kids)
    orphans.sort(key=lambda c: by_id[c].get("label") or c)
    ordered_l0.extend(orphans)

    n_l0 = len(ordered_l0)
    slot = max(78.0, (canvas_w - 2 * margin) / max(n_l0, 1))
    for j, cid in enumerate(ordered_l0):
        x = margin + slot * (j + 0.5)
        y = y_l0_a if j % 2 == 0 else y_l0_b
        positions[cid] = (x, y)

    return positions


def build_site_topology(
    site_id: str,
    *,
    monitor_data: dict,
    parse_device_fn,
) -> dict:
    """依 config port_links 建構單一廠區拓樸。"""
    site_cfg = _site_config(site_id)
    if not site_cfg:
        return {"ok": False, "error": f"未知廠區：{site_id}"}

    cfg = _load_topo_config()
    layers = cfg.get("layers") or []
    subnet = site_cfg.get("subnet") or "192.168.3"

    parsed = monitor_data.get("parsed_logs") or [] if isinstance(monitor_data, dict) else []
    metrics = monitor_data.get("metrics") or {} if isinstance(monitor_data, dict) else {}
    summary = {}
    if isinstance(monitor_data, dict):
        try:
            summary = json.loads(monitor_data.get("all_logs_content") or "{}")
        except Exception:
            summary = {}
    file_metas = summary.get("files") or []

    log_by_index: dict[int, dict] = {}
    if site_cfg.get("bind_ot_logs"):
        for meta in file_metas:
            fname = meta.get("file") or ""
            dev, ip = parse_device_fn(fname)
            if not dev:
                continue
            log_by_index[len(log_by_index)] = {
                "device": dev,
                "ip": ip,
                "events": int(meta.get("event_lines") or 0),
                "file": fname,
                "status": _log_health(parsed, dev, ip),
            }

    auto_xy = _auto_layout_site(site_cfg, layers)

    nodes: list[dict] = []
    for tpl in site_cfg.get("nodes") or []:
        host = tpl.get("host")
        ip = f"{subnet}.{host}" if host is not None else "—"
        nid = tpl["id"]
        st = "ok"
        source = "simulated"
        telemetry: dict = {"subnet": subnet}

        log_idx = tpl.get("log_index")
        if log_idx is not None and log_idx in log_by_index:
            ld = log_by_index[log_idx]
            source = "log"
            st = ld["status"]
            ip = ld["ip"] or ip
            label = ld["device"]
            telemetry.update({"syslog_events": ld["events"], "file": ld["file"]})
        else:
            label = tpl.get("label") or nid
            telemetry.update(_default_telemetry(tpl.get("type") or "server", label, host))
            telemetry["sim_syslog"] = True
            if tpl.get("type") == "workstation":
                drift = metrics.get("recipe_audit", {}).get("count", 0)
                if drift > 5000:
                    st = "warn"
                telemetry["config_drift"] = drift
            if tpl.get("type") == "gateway" and metrics.get("malware_defense", {}).get("count", 0) > 0:
                st = "fail"
            if rng_jitter := (hash(nid) % 7):
                if "syslog_events" in telemetry and isinstance(telemetry["syslog_events"], int):
                    telemetry["syslog_events"] += rng_jitter * 11

        ax, ay = auto_xy.get(nid, (float(tpl.get("x") or 0), float(tpl.get("y") or 0)))
        node_out: dict = {
            "id": nid,
            "label": label,
            "type": tpl.get("type") or "server",
            "layer": tpl.get("layer") or "L1",
            "x": float(ax),
            "y": float(ay),
            "ip": ip,
            "status": st,
            "source": source,
            "telemetry": telemetry,
        }
        if tpl.get("icon"):
            node_out["icon"] = str(tpl["icon"]).strip()
        if tpl.get("image"):
            node_out["image"] = str(tpl["image"]).strip()
        nodes.append(node_out)

    custom_layout = load_custom_layout(site_cfg["id"])
    layout_applied = 0
    if custom_layout:
        layout_applied = apply_layout_positions(nodes, custom_layout.get("nodes") or [])

    by_id = {n["id"]: n for n in nodes}
    edges: list[dict] = []
    port_links: list[dict] = []
    for link in site_cfg.get("port_links") or []:
        fr, to = link.get("from"), link.get("to")
        if fr not in by_id or to not in by_id:
            continue
        edge = {
            "from": fr,
            "to": to,
            "kind": link.get("kind") or "ethernet",
            "label": link.get("label") or "",
            "from_port": link.get("from_port") or "",
            "to_port": link.get("to_port") or "",
            "vlan": link.get("vlan"),
        }
        edges.append(edge)
        port_links.append({**edge, "from_label": by_id[fr]["label"], "to_label": by_id[to]["label"]})

    status_rank = {"fail": 3, "warn": 2, "ok": 1, "offline": 0}
    worst = max((status_rank.get(n.get("status"), 0) for n in nodes), default=1)
    site_status = "fail" if worst >= 3 else ("warn" if worst >= 2 else "ok")

    reports = []
    for ev in parsed[:12]:
        reports.append({
            "time": ev.get("time") or "—",
            "severity": ev.get("severity"),
            "device": ev.get("device") or "",
            "ip": ev.get("ip") or "",
            "summary": (ev.get("raw") or ev.get("message") or "")[:160],
            "control": ev.get("key") or "",
            "simulated": False,
        })

    sim_feed = generate_virtual_syslog_feed(nodes, site_id=site_cfg["id"], count=56)
    merged_feed = reports + sim_feed
    if len(merged_feed) > 64:
        merged_feed = merged_feed[:64]

    return {
        "ok": True,
        "site": {
            "id": site_cfg["id"],
            "name": site_cfg.get("name") or site_cfg["id"],
            "status": site_status,
            "subnet": subnet,
            "principle": monitor_data.get("compliance_principle") if isinstance(monitor_data, dict) else "",
        },
        "layers": layers,
        "canvas": {"width": 1280, "height": 760},
        "nodes": nodes,
        "edges": edges,
        "port_links": port_links,
        "sites_available": list_topology_sites(),
        "log_devices": list(log_by_index.values()),
        "metrics_snapshot": {
            k: v.get("count") for k, v in metrics.items() if isinstance(v, dict)
        } if metrics else {},
        "recent_reports": merged_feed,
        "sim_syslog_feed": sim_feed,
        "sim_machine_count": len(nodes),
        "sim_reporting_devices": [
            {"id": n["id"], "label": n["label"], "ip": n["ip"], "type": n["type"], "source": n["source"]}
            for n in nodes
        ],
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "editable": True,
        "custom_layout": custom_layout,
        "layout_applied_count": layout_applied,
        "layout_schema": _LAYOUT_SCHEMA,
    }


_MONITOR_KEYS = (
    "sec_gem_log",
    "recipe_audit",
    "access_control",
    "patch_management",
    "supplier_security",
    "malware_defense",
)


def _default_monitor_metrics() -> dict:
    return {
        k: {"count": 0, "text": "0 SYSLOG", "status": "pass"}
        for k in _MONITOR_KEYS
    }


def _annex_to_monitor_key(annex_id: str) -> str:
    try:
        from code.services.compliance_service import _OT_MONITORED

        meta = _OT_MONITORED.get((annex_id or "").strip())
        if meta:
            return str(meta.get("control_key") or "recipe_audit")
    except Exception:
        pass
    return "recipe_audit"


_MONITOR_CTRL_LABEL = {
    "sec_gem_log": "A.8.24 (Crypto)",
    "recipe_audit": "A.8.19 (Logs)",
    "access_control": "A.5.15 (Access)",
    "patch_management": "A.8.8 (Patch)",
    "supplier_security": "A.5.19 (Supplier)",
    "malware_defense": "A.8.7 (Malware)",
}


def _topology_event_to_parsed(ev: dict, site_id: str) -> dict:
    annex = (ev.get("control") or "A.8.22").strip()
    key = _annex_to_monitor_key(annex)
    sev = int(ev.get("severity") or 5)
    if sev <= 2:
        badge_type, status_text = "fail-bg", "不符合"
    elif sev <= 3:
        badge_type, status_text = "review-bg", "待改善"
    else:
        badge_type, status_text = "pass-bg", "符合"
    device = ev.get("device") or "ot-device"
    raw = ev.get("summary") or ev.get("raw") or ""
    safe_dev = re.sub(r"[^\w\-]+", "_", device)[:48]
    ctrl_id = _MONITOR_CTRL_LABEL.get(key) or (
        annex if annex.startswith("A.") else f"Annex {annex}"
    )
    ts = ev.get("time") or time.strftime("%H:%M:%S")
    if re.match(r"^\d{2}:\d{2}:\d{2}$", str(ts)):
        ts = time.strftime("%b %d ") + ts + " UTC"
    return {
        "time": ts,
        "file": f"topology://{site_id}/{safe_dev}",
        "raw": raw,
        "message": raw,
        "severity": sev,
        "key": key,
        "control": ctrl_id,
        "label": annex,
        "type": badge_type,
        "statusText": status_text,
        "source": "ot_topology",
        "device": device,
        "ip": ev.get("ip") or "",
        "simulated": bool(ev.get("simulated", True)),
    }


def merge_topology_into_monitor(monitor_data: dict | None, site_id: str = "fab-a") -> dict:
    """
    監控戰情室資料匯入：以 OT 拓樸場域（機台 + 虛擬／日誌 syslog）為主要匯入來源，
    並保留 ot/ 目錄既有解析結果作為補強。
    """
    site_cfg = _site_config(site_id) or _site_config("fab-a")
    sid = (site_cfg or {}).get("id") or site_id or "fab-a"
    site_name = (site_cfg or {}).get("name") or sid

    data: dict = dict(monitor_data) if isinstance(monitor_data, dict) else {}
    # 保留 ot/ 掃描的各控制項原文樣本（拓樸合併後仍供 bundle 顯示）
    if isinstance(monitor_data, dict) and monitor_data.get("samples_by_key"):
        data["samples_by_key"] = monitor_data.get("samples_by_key")
    if data.get("error"):
        data = {
            "metrics": _default_monitor_metrics(),
            "parsed_logs": [],
            "control_bundles": {},
            "all_logs_content": "{}",
            "import_stats": {},
            "compliance_principle": "No Evidence, No Compliance Claim",
        }

    metrics = data.get("metrics") or _default_monitor_metrics()
    for k in _MONITOR_KEYS:
        metrics.setdefault(k, {"count": 0, "text": "0 SYSLOG", "status": "pass"})

    ot_parsed = list(data.get("parsed_logs") or [])
    subnet = (site_cfg or {}).get("subnet") or "192.168.3"
    topo_nodes = [
        {
            "id": tpl["id"],
            "label": tpl.get("label") or tpl["id"],
            "type": tpl.get("type") or "server",
            "ip": f"{subnet}.{tpl.get('host', 1)}",
        }
        for tpl in ((site_cfg or {}).get("nodes") or [])
    ]
    topo_events = generate_virtual_syslog_feed(topo_nodes, site_id=sid, count=72)
    topo_parsed = [_topology_event_to_parsed(ev, sid) for ev in topo_events]

    # 拓樸匯入優先（戰情室「資料匯入」主來源）
    combined: list[dict] = []
    seen_sig: set[tuple] = set()

    def _push(item: dict) -> None:
        sig = (item.get("time"), item.get("raw"), item.get("file"))
        if sig in seen_sig:
            return
        seen_sig.add(sig)
        combined.append(item)

    for item in topo_parsed:
        _push(item)
    for item in ot_parsed:
        if (item.get("source") or "") == "ot_topology":
            continue
        _push(item)

    display_cap = 180
    data["parsed_logs"] = combined[:display_cap]

    # 依拓樸事件更新 KPI 計數（在 ot/ 計數上累加）
    key_counts: dict[str, int] = {k: 0 for k in _MONITOR_KEYS}
    for item in topo_parsed:
        k = item.get("key")
        if k in key_counts:
            key_counts[k] += 1
    for k in _MONITOR_KEYS:
        base = int((metrics.get(k) or {}).get("count") or 0)
        total = base + key_counts.get(k, 0)
        metrics[k]["count"] = total
        metrics[k]["text"] = f"{total} SYSLOG"
        if k == "access_control":
            metrics[k]["status"] = "fail" if total > 8000 else ("review" if total > 2000 else "pass")
        elif k == "malware_defense":
            metrics[k]["status"] = "fail" if total > 0 else "pass"
        elif k in ("recipe_audit", "sec_gem_log", "supplier_security"):
            metrics[k]["status"] = "review" if total > 2000 else "pass"
        elif k == "patch_management":
            metrics[k]["status"] = "review" if total > 100 else "pass"
    data["metrics"] = metrics

    devices = [
        {
            "id": tpl["id"],
            "label": tpl.get("label") or tpl["id"],
            "ip": f"{(site_cfg or {}).get('subnet', '192.168.3')}.{tpl.get('host', 1)}",
        }
        for tpl in ((site_cfg or {}).get("nodes") or [])
    ]
    per_device_lines = max(1, len(topo_parsed) // max(1, len(devices)))
    topo_files = [
        {
            "file": f"topology://{sid}/{d['id']}",
            "format": "topology_syslog",
            "event_lines": per_device_lines,
            "device": d["label"],
            "ip": d["ip"],
        }
        for d in devices
    ]

    ot_stats = data.get("import_stats") or {}
    ot_files = list(ot_stats.get("files") or [])
    ot_event_lines = int(ot_stats.get("total_event_lines") or 0)
    topo_event_lines = len(topo_parsed)

    import_stats = {
        "source": "ot_topology",
        "site_id": sid,
        "site_name": site_name,
        "topology_machines": len(devices),
        "topology_event_lines": topo_event_lines,
        "ot_event_lines": ot_event_lines,
        "total_event_lines": topo_event_lines + ot_event_lines,
        "num_files": len(topo_files) + len(ot_files),
        "file_count": len(topo_files) + len(ot_files),
        "display_sample_cap": display_cap,
        "files": topo_files + ot_files,
    }
    data["import_stats"] = import_stats
    data["topology_import"] = {
        "site_id": sid,
        "site_name": site_name,
        "machines": len(devices),
        "events": topo_event_lines,
        "endpoint": f"/api/ot/topology?site={sid}",
    }
    data.pop("error", None)

    summary_obj = {
        "source": "ot_topology+ot_folder",
        "topology_site": sid,
        "topology_site_name": site_name,
        "topology_machines": len(devices),
        "folder": "ot/",
        "num_files": import_stats["num_files"],
        "total_event_lines": import_stats["total_event_lines"],
        "counts": {k: metrics[k]["count"] for k in _MONITOR_KEYS},
        "import_stats": import_stats,
        "files": import_stats["files"],
    }
    data["all_logs_content"] = json.dumps(summary_obj, ensure_ascii=False, indent=2)
    return data
