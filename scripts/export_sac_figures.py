#!/usr/bin/env python3
"""Export Figure 1 / Figure 2 architecture diagrams (PNG/SVG) for SAC manuscript."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "eval_sac" / "figures"

# Semi-Shield palette
C_BG = "#f8fafc"
C_INGEST = ("#dbeafe", "#2563eb")  # fill, edge
C_ASSESS = ("#e0e7ff", "#4f46e5")
C_RELEASE = ("#d1fae5", "#059669")
C_CORG = ("#fef3c7", "#d97706")
C_TEXT = "#0f172a"
C_MUTED = "#64748b"
C_ARROW = "#475569"


def _setup_style():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Microsoft JhengHei", "Arial", "DejaVu Sans"],
            "figure.facecolor": C_BG,
            "axes.facecolor": C_BG,
            "savefig.facecolor": "white",
            "savefig.dpi": 200,
        }
    )


def _draw_box(ax, x, y, w, h, label, fill, edge, *, fontsize=9, sub=None):
    from matplotlib.patches import FancyBboxPatch

    shadow = FancyBboxPatch(
        (x + 0.03, y - 0.03),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.08",
        facecolor="#cbd5e1",
        edgecolor="none",
        alpha=0.35,
        zorder=1,
    )
    ax.add_patch(shadow)
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.08",
        facecolor=fill,
        edgecolor=edge,
        linewidth=1.8,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2 + (0.08 if sub else 0),
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="600",
        color=C_TEXT,
        zorder=3,
    )
    if sub:
        ax.text(
            x + w / 2,
            y + h / 2 - 0.22,
            sub,
            ha="center",
            va="center",
            fontsize=7,
            color=C_MUTED,
            zorder=3,
        )


def _arrow_h(ax, x1, y, x2, color=C_ARROW):
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(
        FancyArrowPatch(
            (x1, y),
            (x2, y),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.6,
            color=color,
            shrinkA=2,
            shrinkB=2,
            zorder=4,
        )
    )


def _phase_band(ax, x, w, label, color):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            (x, 2.65),
            w,
            0.55,
            boxstyle="round,pad=0.01,rounding_size=0.06",
            facecolor=color,
            edgecolor="none",
            alpha=0.55,
            zorder=0,
        )
    )
    ax.text(x + w / 2, 2.92, label, ha="center", va="center", fontsize=8, color=C_MUTED, fontweight="600")


def fig1():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13.5, 4.2))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0.4, 3.35)
    ax.axis("off")

    ax.text(
        6.75,
        3.18,
        "Figure 1: Semi-Shield assessment and release boundary",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="700",
        color=C_TEXT,
    )
    ax.text(
        6.75,
        0.55,
        "Every revision returns to verification; publication requires an unchanged approved snapshot.",
        ha="center",
        va="center",
        fontsize=8.5,
        color=C_MUTED,
        style="italic",
    )

    _phase_band(ax, 0.35, 4.2, "Ingest & authority", C_INGEST[0])
    _phase_band(ax, 4.65, 4.5, "Assess & CORG", C_ASSESS[0])
    _phase_band(ax, 9.25, 3.9, "Approve & release", C_RELEASE[0])

    # (x, y, w, h, label, (fill, edge), sub)
    nodes = [
        (0.45, 1.35, 1.35, 0.95, "Collector", C_INGEST, "syslog / import"),
        (2.05, 1.35, 1.45, 0.95, "Authority view", C_INGEST, "versioned store"),
        (3.75, 1.35, 1.25, 0.95, "Retrieval", C_INGEST, "eligible bundle"),
        (5.25, 1.35, 1.35, 0.95, "Auditor draft", C_ASSESS, "local LLM"),
        (6.85, 1.35, 1.45, 0.95, "CORG + cert", C_CORG, "closure π, R"),
        (8.55, 1.35, 1.35, 0.95, "Claim checks", C_ASSESS, "verifier"),
        (10.05, 1.35, 1.35, 0.95, "Human approve", C_RELEASE, "one-use token"),
        (11.65, 1.35, 1.35, 0.95, "Release gate", C_RELEASE, "serializable txn"),
    ]

    y_mid = 1.35 + 0.475
    for i, (x, y, w, h, label, pal, sub) in enumerate(nodes):
        _draw_box(ax, x, y, w, h, label, pal[0], pal[1], fontsize=8.5, sub=sub)
        if i < len(nodes) - 1:
            nx = nodes[i + 1][0]
            _arrow_h(ax, x + w + 0.05, y_mid, nx - 0.05)

    # Feedback loop (revision → verify)
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(
        FancyArrowPatch(
            (7.55, 1.28),
            (5.9, 0.78),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color="#94a3b8",
            linestyle="--",
            connectionstyle="arc3,rad=-0.25",
            zorder=4,
        )
    )
    ax.text(6.7, 0.72, "revision / fallback", ha="center", fontsize=7, color=C_MUTED)

    fig.savefig(OUT / "figure1_architecture.png", dpi=200, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(OUT / "figure1_architecture.svg", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def fig2():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0.3, 4.6)
    ax.axis("off")

    ax.text(
        5.5,
        4.35,
        "Figure 2: CORG binds publication to obligation-derived predicate reads",
        ha="center",
        fontsize=12,
        fontweight="700",
        color=C_TEXT,
    )
    ax.text(
        5.5,
        4.05,
        "Including empty result sets, coverage facts, and release-time requery of Sigma_r",
        ha="center",
        fontsize=8.5,
        color=C_MUTED,
    )

    ax.add_patch(
        FancyBboxPatch(
            (0.4, 0.55),
            10.2,
            3.25,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor="#ffffff",
            edgecolor="#cbd5e1",
            linewidth=1.6,
            linestyle="--",
            zorder=0,
        )
    )
    ax.text(0.65, 3.62, "Publication transaction boundary", fontsize=7.5, color=C_MUTED, fontweight="600")

    # Top row: obligation queries
    top = [
        (0.75, 2.55, 2.0, 0.9, "Selectors Sigma_r", C_ASSESS, "closure queries"),
        (3.05, 2.55, 2.0, 0.9, "Empty results", C_INGEST, "authorized empty"),
        (5.35, 2.55, 2.0, 0.9, "Coverage & finality", C_RELEASE, "heartbeats / F_sigma"),
    ]
    for x, y, w, h, label, pal, sub in top:
        _draw_box(ax, x, y, w, h, label, pal[0], pal[1], fontsize=8.5, sub=sub)

    # Center: CORG merges read set
    _draw_box(ax, 3.35, 1.15, 4.3, 1.05, "CORG proof certificate", C_CORG[0], C_CORG[1], fontsize=10, sub="predicate read set R + digest Fa")

    # Bottom row: approval / release
    bot = [
        (0.75, 0.72, 2.35, 0.85, "Approve digest Da", C_CORG, "H(report, q0, Fa)"),
        (8.0, 0.72, 2.35, 0.85, "Release requery Fr", C_RELEASE, "at snapshot tau_r"),
    ]
    for x, y, w, h, label, pal, sub in bot:
        _draw_box(ax, x, y, w, h, label, pal[0], pal[1], fontsize=8.5, sub=sub)

    y_top = 2.55 + 0.45
    y_corg_top = 1.15 + 1.05
    y_corg_bot = 1.15
    for x, _, w, _, *_ in top:
        cx = x + w / 2
        ax.add_patch(
            FancyArrowPatch(
                (cx, y_top),
                (cx, y_corg_top + 0.02),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.5,
                color=C_ARROW,
                zorder=4,
            )
        )

    ax.add_patch(
        FancyArrowPatch(
            (3.8, y_corg_bot - 0.02),
            (1.92, 0.72 + 0.85 + 0.02),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.5,
            color=C_ARROW,
            connectionstyle="arc3,rad=0.15",
            zorder=4,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (7.2, y_corg_bot - 0.02),
            (9.18, 0.72 + 0.85 + 0.02),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.5,
            color=C_ARROW,
            connectionstyle="arc3,rad=-0.15",
            zorder=4,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (2.0, 0.72 + 0.85 / 2),
            (8.0, 0.72 + 0.85 / 2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color="#94a3b8",
            linestyle="--",
            zorder=4,
        )
    )
    ax.text(5.5, 1.0, "bind one-use approval", ha="center", fontsize=7, color=C_MUTED)

    fig.savefig(OUT / "figure2_corg_release.png", dpi=200, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(OUT / "figure2_corg_release.svg", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main() -> int:
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        print("pip install matplotlib", file=sys.stderr)
        return 1

    _setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    fig1()
    fig2()
    print(f"Wrote {OUT / 'figure1_architecture.png'} (+ .svg)")
    print(f"Wrote {OUT / 'figure2_corg_release.png'} (+ .svg)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
