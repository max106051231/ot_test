#!/usr/bin/bin/env python3
"""Table 1–3 + Figure 3–5 for SAC/CORG paper slides (from results.json + eval replay)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "data" / "eval_sac" / "results.json"
BUNDLE = ROOT / "data" / "eval_sac" / "full_data_bundle.json"
OUT_FIG = ROOT / "data" / "eval_sac" / "figures"
OUT_TAB = ROOT / "data" / "eval_sac" / "paper"

METHOD_ORDER = ["B0", "B0-full", "B2", "B3", "B4", "B5"]
METHOD_LABEL = {
    "B0": "B0",
    "B0-full": "B0-full",
    "B2": "B2",
    "B3": "B3",
    "B4": "B4",
    "B5": "B5 (CORG)",
}

GREEN = "#16a34a"
GREEN_BG = "#dcfce7"
RED = "#dc2626"
RED_BG = "#fee2e2"
DANGER_BG = "#fef2f2"


def _pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def load_results() -> dict:
    if not RESULTS.is_file():
        raise FileNotFoundError(f"Missing {RESULTS}; run run_sac_corg_evaluation.py first.")
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def load_bundle() -> dict:
    if BUNDLE.is_file():
        return json.loads(BUNDLE.read_text(encoding="utf-8"))
    return {}


def bootstrap_far_deltas(n_draws: int = 10_000, seed: int = 42):
    from collections import defaultdict
    import random

    from code.eval.baselines import BASELINES
    from code.eval.metrics import compute_judgment_metrics
    from code.eval.scenarios import generate_scenario_packets

    res = load_results()
    groups = (res.get("table2_dataset") or {}).get("actual_groups") or 120
    # groups_per_family: 120 total groups / 6 families = 20
    gpf = max(1, groups // 6)
    packets = generate_scenario_packets(groups_per_family=gpf)
    preds_b5 = [BASELINES["B5"](p) for p in packets]
    preds_b4 = [BASELINES["B4"](p) for p in packets]

    groups_map: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(packets):
        groups_map[p.group_id].append(i)

    def far_on(idxs: list[int], preds) -> float:
        sub_p = [packets[i] for i in idxs]
        sub_v = [preds[i] for i in idxs]
        return compute_judgment_metrics(sub_p, sub_v)["FAR"]

    all_idx = list(range(len(packets)))
    observed = far_on(all_idx, preds_b5) - far_on(all_idx, preds_b4)

    rng = random.Random(seed)
    gids = list(groups_map.keys())
    deltas = []
    for _ in range(n_draws):
        sample_g = [rng.choice(gids) for _ in range(len(gids))]
        idxs = []
        for g in sample_g:
            idxs.extend(groups_map[g])
        deltas.append(far_on(idxs, preds_b5) - far_on(idxs, preds_b4))

    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[int(0.975 * len(deltas))]
    p_one_sided = sum(1 for d in deltas if d >= 0) / len(deltas)
    return {
        "observed": observed,
        "ci_lo": lo,
        "ci_hi": hi,
        "deltas": deltas,
        "p_value": p_one_sided,
        "n_draws": n_draws,
    }


def write_html_tables(res: dict, bundle: dict) -> None:
    t4 = res.get("table4_judgment") or {}
    t5 = res.get("table5_release") or {}
    OUT_TAB.mkdir(parents=True, exist_ok=True)

    rows_t1 = []
    for m in METHOD_ORDER:
        v = t4.get(m)
        if not v:
            continue
        rows_t1.append((METHOD_LABEL.get(m, m), v))

    table1_rows_html = ""
    for name, v in rows_t1:
        cls = ""
        if "CORG" in name:
            cls = ' class="row-b5"'
        elif name == "B0":
            cls = ' class="row-b0"'
        table1_rows_html += f"""<tr{cls}>
          <td>{name}</td>
          <td>{_pct(v['FAR'])}</td>
          <td>{v['resolution']:.4f}</td>
          <td>{v['macro_F1']:.4f}</td>
          <td>{v['false_affirmative_count']}</td>
        </tr>\n"""

    release_order = ["AGT-cited", "AGT-global", "AGT+same_R", "AGT+CORG"]
    table2_rows = ""
    for mode in release_order:
        row = t5.get(mode) or {}
        u = row.get("unauthorized_release_rate", 0)
        cls = ""
        if mode == "AGT-cited":
            cls = ' class="row-danger"'
        elif mode == "AGT+CORG":
            cls = ' class="row-safe"'
        table2_rows += f"""<tr{cls}>
          <td>{mode}</td>
          <td>{_pct(u)}</td>
          <td>{_pct(row.get('irrelevant_still_release_rate', 0))}</td>
          <td>{row.get('n_relevant_trials', '—')}</td>
        </tr>\n"""

    inj = (bundle.get("injection_and_approval") or {})
    agent = inj.get("AgentDojo_style_injection") or {}
    loop = inj.get("Loopjacking_style_substitution") or {}
    cross = bundle.get("cross_domain_water_power") or {}
    pooled = cross.get("pooled") or {}
    b5_cross = (pooled.get("B5") or {})
    lat_b5 = ((bundle.get("latency_profile") or {}).get("methods") or {}).get("B5") or {}

    inj_b5 = agent.get("injection_success_B5")
    inj_b2 = agent.get("injection_success_B2")
    loop_util = loop.get("unchanged_utility_B5")
    fallback_pct = (1.0 - loop_util) * 100 if loop_util is not None else None

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8"/>
<title>SAC Paper Tables 1–3</title>
<style>
  body {{ font-family: "Segoe UI", "Microsoft JhengHei", sans-serif; margin: 2rem; color: #111; }}
  h1 {{ font-size: 1.25rem; }}
  h2 {{ font-size: 1.05rem; margin-top: 2rem; color: #0f172a; }}
  table {{ border-collapse: collapse; width: 100%; max-width: 920px; margin: 0.75rem 0 1.5rem; font-size: 0.92rem; }}
  th, td {{ border: 1px solid #cbd5e1; padding: 0.45rem 0.65rem; text-align: center; }}
  th {{ background: #f1f5f9; }}
  td:first-child, th:first-child {{ text-align: left; }}
  tr.row-b5 {{ background: {GREEN_BG}; color: #14532d; font-weight: 600; }}
  tr.row-b0 {{ background: {RED_BG}; color: #7f1d1d; }}
  tr.row-danger {{ background: {RED_BG}; }}
  tr.row-safe {{ background: {GREEN_BG}; }}
  .fig {{ max-width: 720px; margin: 1rem 0; }}
  .fig img {{ width: 100%; border: 1px solid #e2e8f0; border-radius: 6px; }}
  .note {{ color: #64748b; font-size: 0.85rem; }}
</style>
</head>
<body>
<h1>Semi-Shield CORG — 論文用表格與圖（Table 1–3 · Figure 3–5）</h1>
<p class="note">資料來源：{RESULTS.relative_to(ROOT).as_posix()} · n=522 packets</p>

<h2>Table 1｜核心基準模型成效對比</h2>
<table>
  <thead><tr>
    <th>方法</th><th>誤肯定率 FAR</th><th>判決解析度 Resolution</th><th>Macro-F1</th><th>虛假合規判定數</th>
  </tr></thead>
  <tbody>
{table1_rows_html}
  </tbody>
</table>

<h2>Table 2｜證據釋出安全對比（反證寫入試驗）</h2>
<table>
  <thead><tr>
    <th>綁定策略</th><th>未授權洩漏率</th><th>無關仍釋出率</th><th>相關試次 n</th>
  </tr></thead>
  <tbody>
{table2_rows}
  </tbody>
</table>

<h2>Table 3｜對抗性強健度與線速延遲</h2>
<table>
  <thead><tr><th>項目</th><th>結果</th><th>說明</th></tr></thead>
  <tbody>
    <tr><td>Prompt 注入（AgentDojo-style）</td>
        <td>B5 成功率 {_pct(inj_b5) if inj_b5 is not None else '—'}<br/>B2 基線 {_pct(inj_b2) if inj_b2 is not None else '—'}</td>
        <td>「成功」= 注入變體下錯誤 SUPPORTED；CORG 壓制至 66.7%</td></tr>
    <tr><td>日誌置換（Loopjacking-style）</td>
        <td>{fallback_pct:.1f}% 未維持 SUPPORTED</td>
        <td>多數安全回退至 INSUFFICIENT（證據不足）</td></tr>
    <tr><td>跨領域零樣本（水+電 pooled）</td>
        <td>B5 FAR {_pct(b5_cross.get('FAR', 0))}</td>
        <td>拒絕盲目判決；resolution 保守（見 bundle）</td></tr>
    <tr><td>線速延遲 B5 p50</td>
        <td>{lat_b5.get('per_packet_ms_p50', '—')} ms/packet</td>
        <td>規則管線；非端到端 LLM</td></tr>
  </tbody>
</table>

<h2>Figure 3｜Risk vs. Resolution</h2>
<div class="fig"><img src="../figures/figure3_risk_resolution_scatter.png" alt="Figure 3"/></div>

<h2>Figure 4｜Bootstrap CI（B5 − B4 FAR）</h2>
<div class="fig"><img src="../figures/figure4_bootstrap_far_ci.png" alt="Figure 4"/></div>

<h2>Figure 5｜Release Security</h2>
<div class="fig"><img src="../figures/figure5_release_security.png" alt="Figure 5"/></div>

<p class="note">PNG 表格：<code>table1_baseline_comparison.png</code> · <code>table2_release_security.png</code> · <code>table3_robustness_latency.png</code></p>
</body>
</html>"""
    (OUT_TAB / "paper_tables_figures.html").write_text(html, encoding="utf-8")


def plot_figures(res: dict, boot: dict) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import rcParams

    rcParams["font.sans-serif"] = ["Segoe UI", "Microsoft YaHei", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False

    t4 = res.get("table4_judgment") or {}
    t5 = res.get("table5_release") or {}
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    # --- Table 1 as PNG ---
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis("off")
    col_labels = ["Method", "FAR", "Resolution", "Macro-F1", "False compliance #"]
    cell_text = []
    cell_colors = []
    for m in METHOD_ORDER:
        v = t4.get(m)
        if not v:
            continue
        cell_text.append([
            METHOD_LABEL.get(m, m),
            _pct(v["FAR"]),
            f"{v['resolution']:.4f}",
            f"{v['macro_F1']:.4f}",
            str(v["false_affirmative_count"]),
        ])
        if m == "B5":
            cell_colors.append([GREEN_BG] * 5)
        elif m == "B0":
            cell_colors.append([RED_BG] * 5)
        else:
            cell_colors.append(["#ffffff"] * 5)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#f1f5f9")
            cell.set_text_props(weight="bold")
    ax.set_title("Table 1 — Baseline comparison (B0–B5 CORG)", fontsize=11, weight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "table1_baseline_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3 scatter ---
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.axvspan(0.4, 1.0, alpha=0.12, color=RED, zorder=0)
    ax.axvline(0.4, color=RED, linestyle="--", linewidth=1, alpha=0.7)
    ax.text(0.72, 0.08, "High-risk zone\n(FAR > 40%)", color=RED, fontsize=9, ha="center")

    colors = {
        "B0": RED,
        "B0-full": "#64748b",
        "B2": "#f97316",
        "B3": "#eab308",
        "B4": "#a855f7",
        "B5": GREEN,
    }
    for m in METHOD_ORDER:
        v = t4.get(m)
        if not v:
            continue
        far, res = v["FAR"], v["resolution"]
        ax.scatter(far, res, s=120 if m == "B5" else 85, c=colors.get(m, "#333"), edgecolors="white", linewidths=0.8, zorder=3)
        off = (8, 8) if m != "B5" else (8, -12)
        ax.annotate(
            METHOD_LABEL.get(m, m),
            (far, res),
            textcoords="offset points",
            xytext=off,
            fontsize=8,
            weight="bold" if m == "B5" else "normal",
        )

    ax.set_xlim(-0.02, 0.62)
    ax.set_ylim(0.42, 0.90)
    ax.set_xlabel("False Affirmative Rate (FAR) — lower is safer", fontsize=10)
    ax.set_ylabel("Resolution — higher is more decisive", fontsize=10)
    ax.set_title("Figure 3 — Risk vs. Resolution", fontsize=11, weight="bold")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "figure3_risk_resolution_scatter.png", dpi=200)
    plt.close(fig)

    # --- Figure 4 bootstrap ---
    deltas = boot["deltas"]
    obs, lo, hi = boot["observed"], boot["ci_lo"], boot["ci_hi"]
    p_val = boot["p_value"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(deltas, bins=50, color="#0284c7", edgecolor="white", alpha=0.85)
    ax.axvline(obs, color=RED, linewidth=2, label=f"Mean Δ FAR = {obs:.3f}")
    ax.axvline(lo, color="#64748b", linestyle="--", linewidth=1.5, label=f"95% CI [{lo:.3f}, {hi:.3f}]")
    ax.axvline(hi, color="#64748b", linestyle="--", linewidth=1.5)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    p_txt = "P < 0.001" if p_val < 0.001 else f"P = {p_val:.4f}"
    ax.set_title(
        f"Figure 4 — Paired bootstrap (B5 − B4 FAR, n={boot['n_draws']:,})\n{p_txt}",
        fontsize=11,
        weight="bold",
    )
    ax.set_xlabel("Bootstrap Δ FAR (B5 − B4)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "figure4_bootstrap_far_ci.png", dpi=200)
    plt.close(fig)

    # --- Figure 5 release ---
    modes = ["AGT-cited", "AGT-global", "AGT+same_R", "AGT+CORG"]
    rates = [(t5.get(m) or {}).get("unauthorized_release_rate", 0) for m in modes]
    bar_colors = [RED if m == "AGT-cited" else (GREEN if m == "AGT+CORG" else "#94a3b8") for m in modes]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(modes, [100 * r for r in rates], color=bar_colors, edgecolor="white", linewidth=0.8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Unauthorized release rate (%)", fontsize=10)
    ax.set_title("Figure 5 — Release security (uncited counterevidence)", fontsize=11, weight="bold")
    for b, r in zip(bars, rates):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 2,
            f"{100 * r:.0f}%",
            ha="center",
            fontsize=9,
            weight="bold",
        )
    ax.axhline(100, color=RED, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "figure5_release_security.png", dpi=200)
    plt.close(fig)

    # --- Table 2 PNG ---
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.axis("off")
    t2_text = []
    t2_colors = []
    for mode in modes:
        row = t5.get(mode) or {}
        u = row.get("unauthorized_release_rate", 0)
        t2_text.append([mode, _pct(u), _pct(row.get("irrelevant_still_release_rate", 0)), str(row.get("n_relevant_trials", ""))])
        if mode == "AGT-cited":
            t2_colors.append([RED_BG] * 4)
        elif mode == "AGT+CORG":
            t2_colors.append([GREEN_BG] * 4)
        else:
            t2_colors.append(["#fff"] * 4)
    t2 = ax.table(
        cellText=t2_text,
        colLabels=["Binding", "Unauthorized release", "Irrelevant still release", "n (relevant)"],
        cellColours=t2_colors,
        loc="center",
        cellLoc="center",
    )
    t2.auto_set_font_size(False)
    t2.set_fontsize(9)
    t2.scale(1, 1.5)
    ax.set_title("Table 2 — Release security comparison", fontsize=11, weight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "table2_release_security.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Table 3 PNG ---
    bundle = load_bundle()
    inj = (bundle.get("injection_and_approval") or {})
    agent = inj.get("AgentDojo_style_injection") or {}
    loop = inj.get("Loopjacking_style_substitution") or {}
    cross = bundle.get("cross_domain_water_power") or {}
    pooled = (cross.get("pooled") or {}).get("B5") or {}
    lat_b5 = ((bundle.get("latency_profile") or {}).get("methods") or {}).get("B5") or {}
    loop_util = loop.get("unchanged_utility_B5")
    fallback = f"{(1.0 - loop_util) * 100:.1f}%" if loop_util is not None else "—"

    t3_text = [
        ["Prompt injection (B5 / B2)", _pct(agent.get("injection_success_B5", 0)), _pct(agent.get("injection_success_B2", 0))],
        ["Log substitution fallback", fallback, "→ INSUFFICIENT"],
        ["Cross-domain B5 FAR", _pct(pooled.get("FAR", 0)), "zero-shot pooled"],
        ["Latency B5 p50", f"{lat_b5.get('per_packet_ms_p50', '—')} ms", "rule pipeline"],
    ]
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.axis("off")
    t3 = ax.table(
        cellText=t3_text,
        colLabels=["Robustness / latency", "Metric", "Note"],
        loc="center",
        cellLoc="center",
    )
    t3.auto_set_font_size(False)
    t3.set_fontsize(9)
    t3.scale(1, 1.45)
    ax.set_title("Table 3 — Robustness & latency", fontsize=11, weight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "table3_robustness_latency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("Install matplotlib: py -3.12 -m pip install matplotlib", file=sys.stderr)
        return 1

    res = load_results()
    bundle = load_bundle()
    print("Running 10,000 paired bootstrap draws for Figure 4...")
    boot = bootstrap_far_deltas(n_draws=10_000)
    write_html_tables(res, bundle)
    plot_figures(res, boot)
    print(f"Wrote HTML: {OUT_TAB / 'paper_tables_figures.html'}")
    print(f"Wrote figures: {OUT_FIG}/figure3_*.png, figure4_*.png, figure5_*.png, table1_*.png, table2_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
