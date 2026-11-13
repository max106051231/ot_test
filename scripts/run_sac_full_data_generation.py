#!/usr/bin/env python3
"""
一次產出 SAC 論文缺口表內所有可自動化數據 → data/eval_sac/full_data_bundle.json

涵蓋：dev/holdout+κ、Ollama B2 子集、公開 probe 試點、注入/approval、跨域 30+30、
latency、核心 eval 重跑、圖表與 Figure 1–2。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.eval.annotation_sim import dual_annotate_packets
from code.eval.cross_domain_scenarios import run_cross_domain_eval
from code.eval.injection_battery import run_injection_battery
from code.eval.latency_profile import profile_baselines
from code.eval.llm_writer_eval import run_ollama_b2_subset
from code.eval.public_probes import run_public_probes
from code.eval.scenarios import generate_scenario_packets, load_manifest_counts, save_manifest
from code.paths import data_dir

OUT = data_dir() / "eval_sac"
BUNDLE = OUT / "full_data_bundle.json"


def _run_core_eval(groups: int) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_sac_corg_evaluation.py"),
        "--groups",
        str(groups),
        "--out",
        str(OUT),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return json.loads((OUT / "results.json").read_text(encoding="utf-8"))


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=20, help="held-out groups per family (20→120 groups)")
    ap.add_argument("--dev-per-family", type=int, default=5, help="development groups per family (5→30)")
    ap.add_argument("--skip-ollama", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)

    # 25 groups/family → 30 dev (g0-4) + 120 held-out (g5-24) when dev_per_family=5
    gen_groups = max(args.groups, args.dev_per_family + args.groups)
    packets = generate_scenario_packets(groups_per_family=gen_groups)
    save_manifest(packets, OUT / "scenario_manifest_full.json")

    core = _run_core_eval(args.groups)

    bundle = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_sec": None,
        "table2_annotation": {
            **dual_annotate_packets(packets, dev_groups_per_family=args.dev_per_family),
            "target_dev_groups": 30,
            "target_held_out_groups": 120,
        },
        "core_eval": core,
        "ollama_b2_writer": None if args.skip_ollama else run_ollama_b2_subset(packets, max_cases=36),
        "public_probes": run_public_probes(),
        "injection_and_approval": run_injection_battery(packets),
        "cross_domain_water_power": run_cross_domain_eval(groups_per_domain=30),
        "latency_profile": profile_baselines(packets[:200]),
        "dataset_counts": load_manifest_counts(packets),
    }
    bundle["runtime_sec"] = round(time.perf_counter() - t0, 3)

    BUNDLE.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {BUNDLE}")

    if not args.skip_plots:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "plot_sac_eval.py")], cwd=str(ROOT), check=False)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "export_sac_figures.py")], cwd=str(ROOT), check=False)

    # Summary markdown for thesis
    ann = bundle["table2_annotation"]["overall"]
    md = OUT / "FULL_DATA_SUMMARY.md"
    md.write_text(
        "\n".join(
            [
                "# SAC 全數據 bundle 摘要",
                "",
                f"- 產生時間：`{bundle['generated_at']}`",
                f"- 耗時：{bundle['runtime_sec']}s",
                "",
                "## 1. Dev / Hold-out + Cohen's κ",
                f"- Overall κ (A vs B)：{ann.get('cohen_kappa_a_vs_b')}",
                f"- Raw agreement：{ann.get('raw_agreement_a_b')}",
                "",
                "## 2. Ollama B2 writer",
                f"- Status：`{bundle['ollama_b2_writer'] and bundle['ollama_b2_writer'].get('status')}`",
                "",
                "## 3. Public probes (pilot, separate from OT gold)",
                *(f"- {k}: {v}" for k, v in bundle["public_probes"].items() if k != "disclaimer"),
                "",
                "## 4. Injection / Loopjacking proxy",
                f"- {json.dumps(bundle['injection_and_approval'], ensure_ascii=False)[:400]}…",
                "",
                "## 5. Cross-domain (30 water + 30 power)",
                f"- Total packets：{bundle['cross_domain_water_power']['total_packets']}",
                "",
                "## 6. Latency + figures",
                f"- Figures：`data/eval_sac/figures/`",
                "",
                "完整 JSON：`full_data_bundle.json`",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
