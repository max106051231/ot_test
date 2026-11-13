"""Semi-Shield 論文 Markdown → Word，排版依桌面 Semi-Shield_ISMS.docx 基本格式。"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "docs" / "Semi-Shield_ISMS_大專研究計畫論文.md"
OUT = ROOT / "docs" / "Semi-Shield_ISMS_大專研究計畫論文.docx"
TEMPLATE_DESKTOP = Path(r"c:\Users\Max\Desktop\Semi-Shield_ISMS.docx")
TEMPLATE_REPO = ROOT / "docs" / "templates" / "Semi-Shield_ISMS.docx"

# 與 Semi-Shield_ISMS.docx 一致：DFKai-SB、16pt 題、14pt 章／摘要、12pt 內文、固定行高 18pt
FONT_CH = "DFKai-SB"
FONT_EN = "Times New Roman"
SZ_TITLE = 16
SZ_HEAD = 14
SZ_BODY = 12
LINE_EXACT_PT = 18
INDENT_FIRST_CM = 0.74

def _require_docx():
    try:
        from docx import Document
        from docx.shared import Pt, Cm
        from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
        from docx.oxml.ns import qn
    except ImportError as e:
        raise SystemExit("請先安裝：py -3.12 -m pip install python-docx") from e
    return Document, Pt, Cm, WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, qn


def _template_path() -> Path:
    if TEMPLATE_DESKTOP.is_file():
        return TEMPLATE_DESKTOP
    if TEMPLATE_REPO.is_file():
        return TEMPLATE_REPO
    raise SystemExit(
        "找不到 Semi-Shield_ISMS.docx 範本。請置於桌面或 docs/templates/"
    )


def _clear_body(doc) -> None:
    body = doc.element.body
    for child in list(body):
        if child.tag.endswith("sectPr"):
            continue
        body.remove(child)


def _set_east_asia(run, qn, name: str) -> None:
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)


def _para_format_body(p, Pt, Cm, WD_LINE_SPACING, *, indent_first: bool = True, align=None):
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    pf.line_spacing = Pt(LINE_EXACT_PT)
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    if indent_first:
        pf.first_line_indent = Cm(INDENT_FIRST_CM)
    if align is not None:
        p.alignment = align
    elif align is None and indent_first:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def _add_run_mixed(p, text: str, Pt, qn, *, size: int, bold: bool = False, font_ch: bool = True):
    parts = re.split(r"(\*\*.+?\*\*|`[^`]+`)", text)
    for part in parts:
        if not part:
            continue
        b = bold
        t = part
        if part.startswith("**") and part.endswith("**"):
            t = part[2:-2]
            b = True
        elif part.startswith("`") and part.endswith("`"):
            t = part[1:-1]
        run = p.add_run(t)
        _set_east_asia(run, qn, FONT_CH if font_ch else FONT_EN)
        run.font.size = Pt(size)
        run.bold = b


def _add_title_line(doc, text: str, Pt, qn, WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, Cm):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    pf.line_spacing = Pt(LINE_EXACT_PT)
    pf.space_after = Pt(6)
    run = p.add_run(text)
    _set_east_asia(run, qn, FONT_CH)
    run.font.size = Pt(SZ_TITLE)
    run.bold = True


def _add_author_block(doc, Pt, qn, WD_LINE_SPACING, Cm, WD_ALIGN_PARAGRAPH):
    for line in (
        "第一作者¹、第二作者²（請依實際姓名修改，12pt）",
        "¹所屬單位學系、²所屬單位學系",
        "¹電子郵件、²電子郵件",
    ):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf = p.paragraph_format
        pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        pf.line_spacing = Pt(15)
        run = p.add_run(line)
        _set_east_asia(run, qn, FONT_CH)
        run.font.size = Pt(SZ_BODY)
    doc.add_paragraph()


def _add_label_head(doc, text: str, Pt, qn, WD_LINE_SPACING, Cm):
    """14pt 粗體（摘要、Abstract）— Normal 樣式。"""
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    p = doc.add_paragraph()
    _para_format_body(p, Pt, Cm, WD_LINE_SPACING, indent_first=False)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    _set_east_asia(run, qn, FONT_CH)
    run.font.size = Pt(SZ_HEAD)
    run.bold = True


def _add_chapter_head(doc, text: str, Pt, qn, WD_LINE_SPACING, Cm):
    """章節名 14pt 粗體 — List Paragraph（如：緒論、文獻探討）。"""
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    try:
        p = doc.add_paragraph(style="List Paragraph")
    except KeyError:
        p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    pf.line_spacing = Pt(LINE_EXACT_PT)
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text.strip())
    _set_east_asia(run, qn, FONT_CH)
    run.font.size = Pt(SZ_HEAD)
    run.bold = True


def _add_subsection_head(doc, text: str, Pt, qn, WD_LINE_SPACING, Cm):
    """小節 12pt 粗體（如 1.1、3.4）— Normal。"""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    pf.line_spacing = Pt(LINE_EXACT_PT)
    run = p.add_run(text)
    _set_east_asia(run, qn, FONT_CH)
    run.font.size = Pt(SZ_BODY)
    run.bold = True


def _add_body_para(doc, text: str, Pt, qn, WD_LINE_SPACING, Cm, *, bullet: bool = False):
    p = doc.add_paragraph()
    _para_format_body(p, Pt, Cm, WD_LINE_SPACING, indent_first=not bullet)
    if bullet:
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.left_indent = Cm(0.74)
    body = f"• {text}" if bullet else text
    _add_run_mixed(p, body, Pt, qn, size=SZ_BODY)


def _add_keywords(doc, text: str, Pt, qn, WD_LINE_SPACING, Cm):
    p = doc.add_paragraph()
    _para_format_body(p, Pt, Cm, WD_LINE_SPACING, indent_first=False)
    if text.startswith("關鍵詞") or text.startswith("關鍵字"):
        label, _, rest = text.partition("：")
        if not rest:
            label, _, rest = text.partition(":")
        r0 = p.add_run(label + "：")
        r0.bold = True
        _set_east_asia(r0, qn, FONT_CH)
        r0.font.size = Pt(SZ_BODY)
        r1 = p.add_run(rest.strip())
        r1.bold = True
        _set_east_asia(r1, qn, FONT_CH)
        r1.font.size = Pt(SZ_BODY)
    else:
        _add_run_mixed(p, text, Pt, qn, size=SZ_BODY, bold=True)


def _chapter_title_from_h2(line: str) -> str:
    """## 一、緒論 → 緒論（與 Semi-Shield_ISMS.docx 章節標一致，不加壹貳參／英文副標）。"""
    m = re.match(r"^##\s*[一二三四五六七八九十]+、(.+)$", line.strip())
    if m:
        return m.group(1).strip()
    return line.lstrip("#").strip()


def _parse_table_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_table_sep(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and bool(re.match(r"^\|[\s\-:|]+\|$", s))


def _add_table(doc, rows: list[list[str]], Pt, qn, WD_LINE_SPACING, Cm):
    if not rows:
        return
    ncols = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=ncols)
    try:
        table.style = "Table Grid"
    except KeyError:
        pass
    for ri, row in enumerate(rows):
        for ci in range(ncols):
            val = row[ci] if ci < len(row) else ""
            cell = table.rows[ri].cells[ci]
            cell.text = ""
            p = cell.paragraphs[0]
            _para_format_body(p, Pt, Cm, WD_LINE_SPACING, indent_first=False)
            run = p.add_run(val)
            _set_east_asia(run, qn, FONT_CH)
            run.font.size = Pt(10)
            run.bold = ri == 0
    doc.add_paragraph()


def _extract_front_matter(lines: list[str]) -> tuple[str, str, str, str]:
    title_cn = ""
    subtitle_cn = ""
    abstract = ""
    keywords = ""
    for i, line in enumerate(lines):
        if line.startswith("# ") and not title_cn:
            title_cn = line[2:].strip()
        elif line.startswith("## ") and not subtitle_cn and i < 8:
            sub = line[3:].strip()
            if "半導體" in sub or "Semi" in sub:
                subtitle_cn = sub
        if line.strip().startswith("**摘要：**"):
            abstract = line.strip().replace("**摘要：**", "").strip()
        if line.strip().startswith("**關鍵詞：**") or line.strip().startswith("**關鍵字：**"):
            keywords = line.strip().replace("**", "")
    if not abstract:
        for line in lines:
            if line.strip().startswith("**摘要：**"):
                abstract = line.strip().replace("**摘要：**", "").strip()
                break
    if not keywords:
        for line in lines:
            if line.strip().startswith("**關鍵詞：**"):
                keywords = line.strip().replace("**", "")
                break
    return title_cn, subtitle_cn, abstract, keywords


def main() -> None:
    Document, Pt, Cm, WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, qn = _require_docx()
    if not MD.is_file():
        raise SystemExit(f"找不到 {MD}")

    tpl = _template_path()
    doc = Document(str(tpl))
    _clear_body(doc)

    lines = MD.read_text(encoding="utf-8").splitlines()
    title_cn, subtitle_cn, abstract, keywords = _extract_front_matter(lines)

    # —— 中文刊頭（Semi-Shield_ISMS.docx）——
    _add_title_line(doc, title_cn or "Semi-Shield ISMS 研究計畫／論文", Pt, qn, WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, Cm)
    if subtitle_cn:
        _add_title_line(doc, subtitle_cn, Pt, qn, WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, Cm)
    _add_author_block(doc, Pt, qn, WD_LINE_SPACING, Cm, WD_ALIGN_PARAGRAPH)

    _add_label_head(doc, "摘要", Pt, qn, WD_LINE_SPACING, Cm)
    _add_body_para(doc, abstract, Pt, qn, WD_LINE_SPACING, Cm)
    doc.add_paragraph()
    if keywords:
        _add_keywords(doc, keywords, Pt, qn, WD_LINE_SPACING, Cm)
    doc.add_paragraph()
    doc.add_paragraph()

    # —— 英文刊頭（範本要求中文論文附英文題目／摘要／關鍵詞）——
    en_title = "Semi-Shield ISMS: AI-Driven Evidence Chain Compliance for Semiconductor OT"
    _add_title_line(doc, en_title, Pt, qn, WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, Cm)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("First Author1*, Second Author2")
    _set_east_asia(run, qn, FONT_EN)
    run.font.size = Pt(SZ_BODY)
    doc.add_paragraph()

    _add_label_head(doc, "Abstract", Pt, qn, WD_LINE_SPACING, Cm)
    en_abs = (
        "This work presents Semi-Shield ISMS, an on-premise platform aligning ISO/IEC 27001:2022 "
        "Annex A controls with OT syslog, topology events, and imported audit artifacts. "
        "Evidence registry with SHA-256 hash chaining and local LLM diagnostics operate under "
        "the principle No Evidence, No Compliance Claim."
    )
    _add_body_para(doc, en_abs, Pt, qn, WD_LINE_SPACING, Cm)
    doc.add_paragraph()
    _add_keywords(doc, "Keywords: ISO 27001; OT security; evidence chain; local LLM; syslog", Pt, qn, WD_LINE_SPACING, Cm)
    doc.add_paragraph()
    doc.add_paragraph()

    # —— 正文：自「一、緒論」起，跳過計畫書重複摘要區 ——
    in_code = False
    table_buf: list[list[str]] = []
    title_done = True
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("# ") or (line.startswith("## ") and i < 10):
            i += 1
            continue
        if line.strip().startswith("**撰寫架構") or line.strip().startswith("**實作專案") or line.strip().startswith("**版本：**"):
            i += 1
            continue
        if line.strip().startswith("**摘要：**") or line.strip().startswith("**關鍵詞：**"):
            i += 1
            continue
        if line.strip() == "---" and i < 15:
            i += 1
            continue

        if line.strip().startswith("```"):
            in_code = not in_code
            if in_code:
                _add_body_para(
                    doc,
                    "（架構圖：請自 Markdown 以 Mermaid 匯出 PNG 後插入。）",
                    Pt, qn, WD_LINE_SPACING, Cm,
                )
            i += 1
            continue
        if in_code:
            i += 1
            continue

        m_img = re.match(r"!\[(.+?)\]\((.+?)\)", line.strip())
        if m_img:
            from docx.shared import Inches

            cap, rel = m_img.group(1), m_img.group(2)
            img = ROOT / "docs" / rel.replace("/", "\\")
            if img.is_file():
                doc.add_picture(str(img), width=Inches(5.8))
                cp = doc.add_paragraph()
                cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = cp.add_run(cap)
                run.italic = True
                _set_east_asia(run, qn, FONT_CH)
                run.font.size = Pt(10)
            i += 1
            continue

        if line.strip().startswith("|"):
            if _is_table_sep(line):
                i += 1
                continue
            table_buf.append(_parse_table_row(line))
            if i + 1 >= len(lines) or not lines[i + 1].strip().startswith("|"):
                _add_table(doc, table_buf, Pt, qn, WD_LINE_SPACING, Cm)
                table_buf = []
            i += 1
            continue
        if table_buf:
            _add_table(doc, table_buf, Pt, qn, WD_LINE_SPACING, Cm)
            table_buf = []

        if line.startswith("## "):
            head = _chapter_title_from_h2(line)
            _add_chapter_head(doc, head, Pt, qn, WD_LINE_SPACING, Cm)
            i += 1
            continue
        if line.startswith("### ") or line.startswith("#### "):
            t = line.lstrip("#").strip()
            _add_subsection_head(doc, t, Pt, qn, WD_LINE_SPACING, Cm)
            i += 1
            continue

        if line.strip() == "---":
            i += 1
            continue

        stripped = line.strip()
        if stripped.startswith("- "):
            _add_body_para(doc, stripped[2:], Pt, qn, WD_LINE_SPACING, Cm, bullet=True)
            i += 1
            continue
        if stripped.startswith("**關鍵詞") or stripped.startswith("**關鍵字"):
            i += 1
            continue
        if stripped:
            _add_body_para(doc, stripped, Pt, qn, WD_LINE_SPACING, Cm)
        i += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT))
    print("OK:", OUT.name)


if __name__ == "__main__":
    main()
