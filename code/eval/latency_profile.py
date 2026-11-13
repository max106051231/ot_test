"""Per-method latency p50/p95 and optional RAM snapshot."""
from __future__ import annotations

import time

from code.eval.baselines import BASELINES
from code.eval.scenarios import generate_scenario_packets


def profile_baselines(packets=None, *, repeat: int = 1) -> dict:
    packets = packets or generate_scenario_packets(groups_per_family=8)
    out = {}
    for name, fn in BASELINES.items():
        times_ms: list[float] = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            for p in packets:
                fn(p)
            times_ms.append((time.perf_counter() - t0) * 1000 / max(1, len(packets)))
        times_ms.sort()
        p50 = times_ms[len(times_ms) // 2]
        p95 = times_ms[int(len(times_ms) * 0.95)] if times_ms else 0
        out[name] = {
            "per_packet_ms_p50": round(p50, 4),
            "per_packet_ms_p95": round(p95, 4),
            "n_packets": len(packets),
        }

    ram_mb = None
    try:
        import psutil

        ram_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
    except ImportError:
        pass

    return {
        "methods": out,
        "process_rss_mb": ram_mb,
        "vram_note": "Use nvidia-smi externally; not queried in-process.",
    }
