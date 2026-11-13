#!/usr/bin/env python3
"""Risk–resolution curve + ablation/latency charts from full_data_bundle.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data" / "eval_sac" / "full_data_bundle.json"
OUT_DIR = ROOT / "data" / "eval_sac" / "figures"


def main() -> int:
    if not BUNDLE.is_file():
        print("Run scripts/run_sac_full_data_generation.py first.", file=sys.stderr)
        return 1
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib: py -3.12 -m pip install matplotlib", file=sys.stderr)
        return 1

    data = json.loads(BUNDLE.read_text(encoding="utf-8"))
    t4 = data.get("core_eval", {}).get("table4_judgment") or data.get("table4_judgment", {})
    ab = data.get("ablations", {})
    lat = data.get("latency_profile", {}).get("methods", {})

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Risk–resolution
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, m in t4.items():
        if method not in ("B0", "B2", "B3", "B4", "B5", "B0-full"):
            continue
        ax.scatter(m.get("FAR", 0), m.get("resolution", 0), label=method, s=80)
    ax.set_xlabel("FAR (false-affirmative rate, lower safer)")
    ax.set_ylabel("Answerable-case resolution (higher better)")
    ax.set_title("Risk–resolution operating points")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p1 = OUT_DIR / "risk_resolution_curve.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # Ablation FAR
    if ab:
        fig, ax = plt.subplots(figsize=(5, 3))
        names = list(ab.keys())
        vals = [ab[k].get("FAR_full", 0) for k in names]
        ax.bar(names, vals, color="#0284c7")
        ax.set_ylabel("FAR (B5 full)")
        ax.set_title("Ablation snapshot (FAR)")
        fig.tight_layout()
        p2 = OUT_DIR / "ablation_far.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)

    # Latency p50
    if lat:
        fig, ax = plt.subplots(figsize=(6, 3))
        methods = list(lat.keys())
        p50 = [lat[m].get("per_packet_ms_p50", 0) for m in methods]
        ax.barh(methods, p50, color="#0ea5e9")
        ax.set_xlabel("ms per packet (p50)")
        ax.set_title("Baseline latency profile")
        fig.tight_layout()
        p3 = OUT_DIR / "latency_p50.png"
        fig.savefig(p3, dpi=150)
        plt.close(fig)

    print(f"Wrote figures under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
