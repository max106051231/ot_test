#!/usr/bin/env python3
"""Scan data/ and emit human-readable 資料總覽 (Markdown + HTML)."""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

BASELINE_ZH = {
    "B0": "B0 — 僅 LLM（無 grounded 證據）",
    "B0-full": "B0-full — 保守基線（一律 INSUFFICIENT）",
    "B2": "B2 — LLM + 平面檢索",
    "B3": "B3 — LLM + 圖檢索",
    "B4": "B4 — Semi-Shield（無 CORG）",
    "B5": "B5 — Semi-Shield + CORG",
}

METRIC_ZH = {
    "FAR": "誤肯定率 FAR（gold 非 SUPPORTED 卻判 SUPPORTED）",
    "resolution": "可回答題解析率",
    "macro_F1": "Macro-F1",
    "false_affirmative_count": "誤肯定筆數",
}


def iter_jsonl(path: Path):
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def load_json(path: Path):
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def summarize_jsonl(path: Path) -> dict:
    n = 0
    return {"path": str(path.relative_to(ROOT)), "lines": n}


def scan_evidence_registry(path: Path) -> dict:
    by_annex = Counter()
    by_status = Counter()
    by_control = Counter()
    total_events = 0
    n = 0
    for row in iter_jsonl(path):
        n += 1
        by_annex[row.get("annex_a_id") or "(未標)"] += 1
        by_status[row.get("review_status") or "unknown"] += 1
        by_control[row.get("control_key") or "unknown"] += 1
        total_events += int(row.get("event_count") or 0)
    return {
        "records": n,
        "by_annex_top": by_annex.most_common(12),
        "by_review_status": dict(by_status),
        "by_control_top": by_control.most_common(10),
        "total_syslog_events": total_events,
    }


def scan_review_queue(path: Path) -> dict:
    by_status = Counter()
    by_priority = Counter()
    by_type = Counter()
    recent: list[dict] = []
    n = 0
    for row in iter_jsonl(path):
        n += 1
        by_status[row.get("status") or "unknown"] += 1
        by_priority[row.get("priority") or "unknown"] += 1
        by_type[row.get("item_type") or "unknown"] += 1
        recent.append(row)
        if len(recent) > 5:
            recent.pop(0)
    return {
        "records": n,
        "by_status": dict(by_status),
        "by_priority": dict(by_priority),
        "by_item_type": dict(by_type),
        "recent": [
            {
                "review_id": r.get("review_id"),
                "title": r.get("title"),
                "status": r.get("status"),
                "priority": r.get("priority"),
                "updated_at": r.get("updated_at"),
            }
            for r in recent
        ],
    }


def scan_manual_imports(path: Path) -> dict:
    by_annex = Counter()
    titles = []
    n = 0
    simulated = 0
    for row in iter_jsonl(path):
        n += 1
        by_annex[row.get("annex_a_id") or "(未標)"] += 1
        if row.get("simulated"):
            simulated += 1
        if len(titles) < 8:
            titles.append(
                {
                    "import_id": row.get("import_id"),
                    "annex_a_id": row.get("annex_a_id"),
                    "title": row.get("title"),
                    "imported_at": row.get("imported_at"),
                }
            )
    return {
        "records": n,
        "simulated_count": simulated,
        "by_annex": dict(by_annex),
        "sample": titles,
    }


def list_data_files() -> list[dict]:
    rows = []
    if not DATA.is_dir():
        return rows
    for p in sorted(DATA.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(DATA).as_posix()
        rows.append(
            {
                "path": rel,
                "size_kb": round(p.stat().st_size / 1024, 1),
                "kind": p.suffix.lower() or "(無副檔名)",
            }
        )
    return rows


def build_overview() -> dict:
    generated = datetime.now(timezone.utc).isoformat()
    files = list_data_files()

    evidence = scan_evidence_registry(DATA / "evidence_registry.jsonl")
    review = scan_review_queue(DATA / "review_queue.jsonl")

    manual = scan_manual_imports(DATA / "manual_imports.jsonl")
    results = load_json(DATA / "eval_sac" / "results.json")
    bundle = load_json(DATA / "eval_sac" / "full_data_bundle.json")

    figures = []
    fig_dir = DATA / "eval_sac" / "figures"
    if fig_dir.is_dir():
        for p in sorted(fig_dir.iterdir()):
            if p.suffix.lower() in {".png", ".svg", ".jfif", ".jpg", ".jpeg"}:
                figures.append(
                    {
                        "name": p.name,
                        "path": p.relative_to(DATA).as_posix(),
                        "path_from_root": p.relative_to(ROOT).as_posix(),
                    }
                )

    return {
        "generated_at": generated,
        "files": files,
        "evidence_registry": evidence,
        "review_queue": review,
        "manual_imports": manual,
        "eval_sac": {"results": results, "bundle": bundle, "figures": figures},
    }


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_markdown(ov: dict) -> str:
    ev = ov["evidence_registry"]
    rq = ov["review_queue"]
    mi = ov["manual_imports"]
    res = ov["eval_sac"]["results"] or {}
    bundle = ov["eval_sac"]["bundle"] or {}

    parts = [
        "# Semi-Shield 資料目錄總覽",
        "",
        f"- 產生時間（UTC）：`{ov['generated_at']}`",
        "- 本文件由 `scripts/build_data_overview.py` 自動產生；原始檔仍在 `data/` 各子目錄。",
        "",
        "## 1. 這個資料夾裡有什麼？",
        "",
        "| 檔案 | 大小 (KB) | 說明 |",
        "| --- | ---: | --- |",
    ]

    desc = {
        "evidence_registry.jsonl": "OT 證據鏈：每行一筆證據（Annex A、syslog 聚合、hash chain）",
        "review_queue.jsonl": "人工／AI 覆核佇列（control 狀態、優先級）",
        "manual_imports.jsonl": "手動匯入的 ISMS 條文／政策片段（示範批次）",
        "eval_sac/results.json": "SAC/CORG 實驗核心表（Table 2/4/5/6 數值）",
        "eval_sac/full_data_bundle.json": "完整實驗 bundle（κ、公開探針、延遲等）",
        "eval_sac/scenario_manifest.json": "評估情境 manifest（held-out 子集）",
        "data.zip": "資料打包備份",
    }
    for f in ov["files"]:
        key = f["path"]
        note = desc.get(key, "—")
        parts.append(f"| `{key}` | {f['size_kb']} | {note} |")

    parts += [
        "",
        "## 2. 證據登錄 `evidence_registry.jsonl`",
        "",
        f"- **總筆數**：{ev['records']}",
        f"- **彙總 syslog 事件數**：{ev['total_syslog_events']:,}",
        "",
        "### 覆核狀態",
        "",
    ]
    for k, v in sorted(ev["by_review_status"].items(), key=lambda x: -x[1]):
        parts.append(f"- `{k}`：{v} 筆")

    parts += ["", "### Annex A 分布（前 12）", ""]
    for annex, cnt in ev["by_annex_top"]:
        parts.append(f"- `{annex}`：{cnt} 筆")

    parts += [
        "",
        "## 3. 覆核佇列 `review_queue.jsonl`",
        "",
        f"- **總筆數**：{rq['records']}",
        "",
        "### 狀態",
        "",
    ]
    for k, v in sorted(rq["by_status"].items(), key=lambda x: -x[1]):
        parts.append(f"- `{k}`：{v} 筆")

    parts += ["", "### 優先級", ""]
    for k, v in sorted(rq["by_priority"].items(), key=lambda x: -x[1]):
        parts.append(f"- `{k}`：{v} 筆")

    parts += [
        "",
        "## 4. 手動匯入 `manual_imports.jsonl`",
        "",
        f"- **總筆數**：{mi['records']}（其中示範 simulated：{mi['simulated_count']}）",
        "",
        "### 依 Annex A",
        "",
    ]
    for k, v in sorted(mi["by_annex"].items()):
        parts.append(f"- `{k}`：{v} 筆")

    if res:
        ds = res.get("table2_dataset") or {}
        gold = ds.get("gold_SVI") or {}
        parts += [
            "",
            "## 5. SAC 實驗結果 `eval_sac/results.json`",
            "",
            f"- **評估封包數**：{ds.get('actual_packets', '—')}",
            f"- **情境組數**：{ds.get('actual_groups', '—')}",
            f"- Gold 標籤：SUPPORTED {gold.get('SUPPORTED', 0)} / VIOLATED {gold.get('VIOLATED', 0)} / INSUFFICIENT {gold.get('INSUFFICIENT', 0)}",
            "",
            "### Table 4 — 各基線判斷品質",
            "",
            md_table(
                ["基線", "FAR", "解析率", "Macro-F1", "誤肯定數"],
                [
                    [
                        BASELINE_ZH.get(k, k),
                        pct(v["FAR"]),
                        pct(v["resolution"]),
                        f"{v['macro_F1']:.4f}",
                        str(v["false_affirmative_count"]),
                    ]
                    for k, v in (res.get("table4_judgment") or {}).items()
                ],
            ),
            "",
            "### Table 5 — 證據釋出（CORG）",
            "",
            md_table(
                ["策略", "未授權釋出率", "無關仍釋出率", "相關試次", "無關試次"],
                [
                    [
                        name,
                        pct(row["unauthorized_release_rate"]),
                        pct(row["irrelevant_still_release_rate"]),
                        str(row["n_relevant_trials"]),
                        str(row["n_irrelevant_trials"]),
                    ]
                    for name, row in (res.get("table5_release") or {}).items()
                ],
            ),
            "",
            "**解讀提示**：B5（含 CORG）在本 run 中 FAR=0，代表 gold 非 SUPPORTED 時不會誤判為 SUPPORTED；AGT-cited 未授權釋出率 100% 對照 AGT+CORG 為 0%。",
        ]

    if bundle:
        ann = bundle.get("table2_annotation") or {}
        overall = ann.get("overall") or {}
        parts += [
            "",
            "## 6. 完整 bundle 摘要 `full_data_bundle.json`",
            "",
            f"- Cohen's κ（模擬雙標者）：**{overall.get('cohen_kappa_a_vs_b', '—')}**",
            f"- 原始一致率：{overall.get('raw_agreement_a_b', '—')}",
            f"- 註：κ 為 pilot 模擬標註，正式論文應換專家複審。",
            "",
            "### 公開探針（與 OT gold 分開）",
            "",
        ]
        probes = bundle.get("public_probes") or {}
        for key, val in probes.items():
            if key == "disclaimer":
                parts.append(f"- _{val}_")
            elif isinstance(val, dict):
                parts.append(f"- **{key}**：{json.dumps(val, ensure_ascii=False)}")

    parts += [
        "",
        "## 7. 圖表",
        "",
    ]
    for fig in ov["eval_sac"]["figures"]:
        parts.append(f"- `{fig.get('path_from_root', fig['path'])}`")

    parts += [
        "",
        "---",
        "在瀏覽器開啟 **`data/data_overview.html`** 可看表格化儀表板（含圖片預覽）。",
        "",
    ]
    return "\n".join(parts)


def render_html(ov: dict) -> str:
    payload = json.dumps(ov, ensure_ascii=False)
    baseline_rows = ""
    res = ov["eval_sac"]["results"] or {}
    for k, v in (res.get("table4_judgment") or {}).items():
        baseline_rows += f"""<tr>
          <td>{BASELINE_ZH.get(k, k)}</td>
          <td>{pct(v['FAR'])}</td>
          <td>{pct(v['resolution'])}</td>
          <td>{v['macro_F1']:.4f}</td>
          <td>{v['false_affirmative_count']}</td>
        </tr>"""

    release_rows = ""
    for name, row in (res.get("table5_release") or {}).items():
        release_rows += f"""<tr>
          <td>{name}</td>
          <td>{pct(row['unauthorized_release_rate'])}</td>
          <td>{pct(row['irrelevant_still_release_rate'])}</td>
          <td>{row['n_relevant_trials']}</td>
          <td>{row['n_irrelevant_trials']}</td>
        </tr>"""

    fig_cards = ""
    for fig in ov["eval_sac"]["figures"]:
        if fig["name"].lower().endswith((".png", ".jpg", ".jpeg", ".jfif")):
            fig_cards += f"""
            <figure class="fig">
              <img src="{fig['path']}" alt="{fig['name']}" loading="lazy" />
              <figcaption>{fig['name']}</figcaption>
            </figure>"""

    ev = ov["evidence_registry"]
    rq = ov["review_queue"]
    mi = ov["manual_imports"]
    ds = (res.get("table2_dataset") or {}) if res else {}
    gold = ds.get("gold_SVI") or {}
    kappa = ""
    bundle = ov["eval_sac"]["bundle"] or {}
    if bundle:
        overall = (bundle.get("table2_annotation") or {}).get("overall") or {}
        kappa = str(overall.get("cohen_kappa_a_vs_b", ""))

    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Semi-Shield 資料總覽</title>
  <style>
    :root {{
      --bg: #0f1419;
      --card: #1a2332;
      --text: #e7ecf3;
      --muted: #9aa8bc;
      --accent: #3d9cf5;
      --border: #2a3548;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Microsoft JhengHei", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.55;
    }}
    header {{
      padding: 1.5rem 2rem;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(135deg, #152238, #0f1419);
    }}
    header h1 {{ margin: 0 0 0.35rem; font-size: 1.5rem; }}
    header p {{ margin: 0; color: var(--muted); font-size: 0.9rem; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 1.5rem 2rem 3rem; }}
    section {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.25rem 1.5rem;
      margin-bottom: 1.25rem;
    }}
    h2 {{ margin-top: 0; font-size: 1.15rem; color: var(--accent); }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 0.75rem;
      margin: 1rem 0;
    }}
    .stat {{
      background: #121a26;
      border-radius: 8px;
      padding: 0.75rem 1rem;
      border: 1px solid var(--border);
    }}
    .stat b {{ display: block; font-size: 1.35rem; }}
    .stat span {{ color: var(--muted); font-size: 0.85rem; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 0.5rem 0.6rem;
      text-align: left;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    .files {{ font-size: 0.85rem; max-height: 220px; overflow: auto; }}
    .fig-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 1rem;
    }}
    .fig img {{
      width: 100%;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #fff;
    }}
    .fig figcaption {{ font-size: 0.8rem; color: var(--muted); margin-top: 0.35rem; }}
    .hint {{ color: var(--muted); font-size: 0.88rem; }}
    ul.compact {{ margin: 0.5rem 0; padding-left: 1.2rem; }}
  </style>
</head>
<body>
  <header>
    <h1>Semi-Shield 資料總覽</h1>
    <p>產生時間（UTC）：{ov['generated_at']} · 對應目錄 <code>data/</code></p>
  </header>
  <main>
    <section>
      <h2>快速摘要</h2>
      <div class="stats">
        <div class="stat"><b>{ev['records']}</b><span>證據登錄筆數</span></div>
        <div class="stat"><b>{ev['total_syslog_events']:,}</b><span>彙總 syslog 事件</span></div>
        <div class="stat"><b>{rq['records']}</b><span>覆核佇列</span></div>
        <div class="stat"><b>{mi['records']}</b><span>手動匯入</span></div>
        <div class="stat"><b>{ds.get('actual_packets', '—')}</b><span>SAC 評估封包</span></div>
        <div class="stat"><b>{kappa or '—'}</b><span>Cohen's κ（pilot）</span></div>
      </div>
      <p class="hint">Gold 標籤（SAC）：SUPPORTED {gold.get('SUPPORTED', 0)} · VIOLATED {gold.get('VIOLATED', 0)} · INSUFFICIENT {gold.get('INSUFFICIENT', 0)}</p>
    </section>

    <section>
      <h2>檔案一覽</h2>
      <div class="files">
        <table>
          <thead><tr><th>路徑</th><th>大小 KB</th><th>類型</th></tr></thead>
          <tbody>
            {''.join(f"<tr><td><code>{f['path']}</code></td><td>{f['size_kb']}</td><td>{f['kind']}</td></tr>" for f in ov['files'])}
          </tbody>
        </table>
      </div>
    </section>

    <section>
      <h2>證據與覆核</h2>
      <p><strong>證據覆核狀態</strong></p>
      <ul class="compact">
        {''.join(f"<li><code>{k}</code>：{v} 筆</li>" for k, v in sorted(ev['by_review_status'].items(), key=lambda x: -x[1]))}
      </ul>
      <p><strong>覆核佇列狀態</strong></p>
      <ul class="compact">
        {''.join(f"<li><code>{k}</code>：{v} 筆</li>" for k, v in sorted(rq['by_status'].items(), key=lambda x: -x[1]))}
      </ul>
      <p><strong>手動匯入（Annex A）</strong></p>
      <ul class="compact">
        {''.join(f"<li><code>{k}</code>：{v} 筆</li>" for k, v in sorted(mi['by_annex'].items()))}
      </ul>
    </section>

    <section>
      <h2>SAC Table 4 — 判斷基線</h2>
      <p class="hint">FAR 越低越好（勿把非合規判成 SUPPORTED）。B5 含 CORG 約束。</p>
      <table>
        <thead>
          <tr><th>基線</th><th>FAR</th><th>解析率</th><th>Macro-F1</th><th>誤肯定數</th></tr>
        </thead>
        <tbody>{baseline_rows or '<tr><td colspan="5">無 results.json</td></tr>'}</tbody>
      </table>
    </section>

    <section>
      <h2>SAC Table 5 — 證據釋出</h2>
      <table>
        <thead>
          <tr><th>策略</th><th>未授權釋出率</th><th>無關仍釋出</th><th>相關 n</th><th>無關 n</th></tr>
        </thead>
        <tbody>{release_rows or '<tr><td colspan="5">—</td></tr>'}</tbody>
      </table>
    </section>

    <section>
      <h2>實驗圖表</h2>
      <div class="fig-grid">{fig_cards or '<p class="hint">無圖檔</p>'}</div>
    </section>
  </main>
  <script type="application/json" id="overview-data">{payload}</script>
</body>
</html>"""


def main():
    ov = build_overview()
    md_path = DATA / "資料總覽.md"
    html_path = DATA / "data_overview.html"
    md_path.write_text(render_markdown(ov), encoding="utf-8")
    html_path.write_text(render_html(ov), encoding="utf-8")
    print("Wrote data/資料總覽.md")
    print("Wrote data/data_overview.html")


if __name__ == "__main__":
    main()
