#!/usr/bin/env python3
"""
執行 SAC CORG 評估協議（可重現子集）：產生 scenario、跑 B0–B5、release 變異、輸出 Table 2/4/5/6。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.eval.baselines import BASELINES
from code.eval.metrics import (
    compute_judgment_metrics,
    paired_bootstrap_ci,
    summarize_release,
)
from code.eval.release_eval import run_release_battery
from code.eval.scenarios import (
    generate_scenario_packets,
    load_manifest_counts,
    save_manifest,
)
from code.paths import data_dir, project_root


def _run_ablation_sample(packets, fn_full, fn_ablated, n=80):
    import random

    rng = random.Random(0)
    idxs = rng.sample(range(len(packets)), min(n, len(packets)))
    sub = [packets[i] for i in idxs]
    full = [fn_full(p) for p in sub]
    ab = [fn_ablated(p) for p in sub]
    m_full = compute_judgment_metrics(sub, full)
    m_ab = compute_judgment_metrics(sub, ab)
    return {"FAR_full": m_full["FAR"], "FAR_ablated": m_ab["FAR"], "n": len(sub)}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=12, help="每 monitoring family 的 scenario groups")
    ap.add_argument("--out", type=Path, default=data_dir() / "eval_sac")
    args = ap.parse_args()

    t0 = time.perf_counter()
    out: Path = args.out if args.out.is_absolute() else project_root() / args.out
    out.mkdir(parents=True, exist_ok=True)

    packets = generate_scenario_packets(groups_per_family=args.groups)
    save_manifest(packets, out / "scenario_manifest.json")

    table2 = load_manifest_counts(packets)
    table2["partition"] = {
        "development_target": 30,
        "held_out_target": 120,
        "note": "本 run 以 ot/ 語料合成 held-out 子集；獨立 dev split 請另 manifest",
    }

    judgment: dict[str, dict] = {}
    preds_by_method: dict[str, list] = {}
    for name, fn in BASELINES.items():
        preds = [fn(p) for p in packets]
        preds_by_method[name] = preds
        judgment[name] = compute_judgment_metrics(packets, preds)

    # RQ1 CI: B5 vs B4 FAR
    delta_far, ci_lo, ci_hi = paired_bootstrap_ci(
        packets,
        preds_by_method["B5"],
        preds_by_method["B4"],
        lambda ps, pr: compute_judgment_metrics(ps, pr)["FAR"],
    )
    # RQ3: B3 vs B2 resolution
    delta_res, res_lo, res_hi = paired_bootstrap_ci(
        packets,
        preds_by_method["B3"],
        preds_by_method["B2"],
        lambda ps, pr: compute_judgment_metrics(ps, pr)["resolution"],
    )

    release_rows = run_release_battery(packets)
    table5 = summarize_release(release_rows)

    # Ablations (disable span / coverage checks in B5 path)
    from code.eval.obligations import OBLIGATIONS, corg_assess

    def b5_no_span(p):
        obl = OBLIGATIONS[p.annex_id]
        v, _, _ = corg_assess(
            obl, p.records, site=p.scope_site, assets=p.scope_assets,
            coverage_ok=p.collector_coverage_ok, require_cited_span=False,
        )
        return v

    def b5_no_coverage(p):
        obl = OBLIGATIONS[p.annex_id]
        v, _, _ = corg_assess(
            obl, p.records, site=p.scope_site, assets=p.scope_assets,
            coverage_ok=True, require_cited_span=True,
        )
        return v

    ablations = {
        "disable_cited_span": _run_ablation_sample(packets, BASELINES["B5"], b5_no_span),
        "ignore_coverage_flag": _run_ablation_sample(packets, BASELINES["B5"], b5_no_coverage),
    }

    table6 = [
        {
            "contrast": "B5 vs B4",
            "metric": "FAR",
            "observed_delta": delta_far,
            "ci_95": [ci_lo, ci_hi],
            "hypothesis": "FAR lower with CORG at comparable resolution",
        },
        {
            "contrast": "B3 vs B2",
            "metric": "resolution",
            "observed_delta": delta_res,
            "ci_95": [res_lo, res_hi],
            "hypothesis": "graph retrieval improves answerable resolution",
        },
        {
            "contrast": "AGT+CORG vs AGT-cited",
            "metric": "unauthorized_release_rate",
            "observed_delta": (
                table5.get("AGT-cited", {}).get("unauthorized_release_rate", 0)
                - table5.get("AGT+CORG", {}).get("unauthorized_release_rate", 0)
            ),
            "note": "positive delta means CORG blocks more unsafe releases",
        },
    ]

    elapsed = time.perf_counter() - t0
    summary = {
        "protocol": "Semi-Shield OT Evidence Evaluation (executable subset)",
        "runtime_sec": round(elapsed, 3),
        "table2_dataset": table2,
        "table4_judgment": judgment,
        "table5_release": table5,
        "table6_interpretation": table6,
        "ablations": ablations,
        "release_detail_count": len(release_rows),
    }

    (out / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md_lines = [
        "# SAC CORG 評估結果（本機執行）",
        "",
        f"- 執行時間：{elapsed:.2f}s",
        f"- Packets：{table2['actual_packets']}（groups={table2['actual_groups']}）",
        "",
        "## Table 2 — Dataset composition",
        "",
        f"| 項目 | 值 |",
        f"|------|-----|",
        f"| actual_groups | {table2['actual_groups']} |",
        f"| actual_packets | {table2['actual_packets']} |",
        f"| SUPPORTED | {table2['gold_SVI']['SUPPORTED']} |",
        f"| VIOLATED | {table2['gold_SVI']['VIOLATED']} |",
        f"| INSUFFICIENT | {table2['gold_SVI']['INSUFFICIENT']} |",
        "",
        "## Table 4 — OT judgment (FAR ↓, resolution ↑)",
        "",
        "| Method | FAR | Resolution | Macro-F1 |",
        "|--------|-----|------------|----------|",
    ]
    for method in ("B0", "B0-full", "B2", "B3", "B4", "B5"):
        m = judgment[method]
        md_lines.append(
            f"| {method} | {m['FAR']:.4f} | {m['resolution']:.4f} | {m['macro_F1']:.4f} |"
        )
    md_lines.extend(
        [
            "",
            "## Table 5 — Release (uncited counterevidence mutation)",
            "",
            "| Binding | Unauthorized release rate | Irrelevant still releases |",
            "|---------|---------------------------|---------------------------|",
        ]
    )
    for mode in ("AGT-cited", "AGT-global", "AGT+CORG", "AGT+same_R"):
        t = table5.get(mode, {})
        md_lines.append(
            f"| {mode} | {t.get('unauthorized_release_rate', '—')} | "
            f"{t.get('irrelevant_still_release_rate', '—')} |"
        )
    md_lines.extend(["", "## Table 6 — Key contrasts", ""])
    for row in table6:
        md_lines.append(f"- **{row['contrast']}** ({row.get('metric', '')}): Δ={row.get('observed_delta')}")
    md_lines.append("")
    md_lines.append(f"完整 JSON：`{out / 'results.json'}`")

    report_path = project_root() / "docs" / "SAC_EVAL_RESULTS.md"
    report_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote {out / 'results.json'}")
    print(f"Wrote {report_path}")
    for method in ("B0", "B2", "B4", "B5"):
        m = judgment[method]
        print(f"{method}: FAR={m['FAR']:.4f} resolution={m['resolution']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
