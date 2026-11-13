"""由 Markdown 論文版與 ppt_assets 截圖產生 Word（.docx）。"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "docs" / "Semi-Shield_AI_Evidence_Chain_論文版.md"
OUT = ROOT / "docs" / "Semi-Shield_AI_Evidence_Chain_論文版.docx"
ASSETS = ROOT / "docs" / "ppt_assets"


def _require_docx():
    try:
        from docx import Document
        from docx.shared import Inches, Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError as e:
        raise SystemExit(
            "請先安裝 python-docx：py -3.12 -m pip install python-docx"
        ) from e
    return Document, Inches, Pt, WD_ALIGN_PARAGRAPH


def _add_heading(doc, text: str, level: int) -> None:
    t = text.strip()
    if not t:
        return
    doc.add_heading(t, level=min(level, 3))


def _add_para(doc, text: str, Pt, *, bold: bool = False) -> None:
    t = text.strip()
    if not t:
        return
    p = doc.add_paragraph()
    run = p.add_run(t)
    run.bold = bold
    run.font.size = Pt(11)


def _add_image(doc, rel_path: str, caption: str, Inches, Pt, WD_ALIGN_PARAGRAPH) -> None:
    img = ROOT / "docs" / rel_path.replace("/", "\\")
    if not img.is_file():
        _add_para(doc, f"[附圖缺失：{rel_path}]", Pt, bold=True)
        return
    doc.add_picture(str(img), width=Inches(6.2))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    r.italic = True
    r.font.size = Pt(10)


def main() -> None:
    Document, Inches, Pt, WD_ALIGN_PARAGRAPH = _require_docx()
    if not MD.is_file():
        raise SystemExit(f"找不到 {MD}")

    text = MD.read_text(encoding="utf-8")
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Microsoft JhengHei"
    style.font.size = Pt(11)

    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            if in_code:
                _add_para(
                    doc,
                    "（架構圖見 Markdown 內 mermaid 區塊；Word 版請以 draw.io 或 Mermaid 匯出 PNG 後插入。）",
                    Pt,
                )
            continue
        if in_code:
            continue

        m_img = re.match(r"!\[(.+?)\]\((.+?)\)", line.strip())
        if m_img:
            caption, path = m_img.group(1), m_img.group(2)
            _add_image(doc, path, caption, Inches, Pt, WD_ALIGN_PARAGRAPH)
            continue

        if line.startswith("# "):
            _add_heading(doc, line[2:], 0)
        elif line.startswith("## "):
            _add_heading(doc, line[3:], 1)
        elif line.startswith("### "):
            _add_heading(doc, line[4:], 2)
        elif line.startswith("|") and "---" not in line:
            _add_para(doc, line.replace("|", " ").strip(), Pt)
        elif line.strip() == "---":
            doc.add_paragraph()
        elif line.strip():
            _add_para(doc, line, Pt)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT))
    print("Wrote:", OUT)


if __name__ == "__main__":
    main()
