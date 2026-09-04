#!/usr/bin/env python3
"""
將 docs/ARCHITECTURE.md 匯出為 PDF（繁體中文）。

用法：
  python scripts/export_architecture_pdf.py
  python scripts/export_architecture_pdf.py -o docs/ARCHITECTURE.pdf
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MD = ROOT / "docs" / "ARCHITECTURE.md"
DEFAULT_PDF = ROOT / "docs" / "ARCHITECTURE.pdf"

FONT_REGULAR = "SemiShieldCJK"
FONT_BOLD = "SemiShieldCJK-Bold"

# (regular, bold)；.ttc 需 subfontIndex=0
FONT_PAIRS: list[tuple[Path, Path | None]] = [
    (Path(r"C:\Windows\Fonts\msjh.ttc"), Path(r"C:\Windows\Fonts\msjhbd.ttc")),
    (Path(r"C:\Windows\Fonts\mingliu.ttc"), Path(r"C:\Windows\Fonts\mingliub.ttc")),
    (Path(r"C:\Windows\Fonts\kaiu.ttf"), None),
    (Path("/System/Library/Fonts/PingFang.ttc"), None),
    (Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"), None),
    (Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"), None),
]


def _load_tt_font(name: str, path: Path) -> None:
    """載入 TTF/TTC；TTC 集合必須指定 subfontIndex=0，否則中文會變方塊／亂碼。"""
    suffix = path.suffix.lower()
    if suffix == ".ttc":
        pdfmetrics.registerFont(TTFont(name, str(path), subfontIndex=0))
    else:
        pdfmetrics.registerFont(TTFont(name, str(path)))


def _register_cjk_font() -> tuple[str, str]:
    for regular_path, bold_path in FONT_PAIRS:
        if not regular_path.is_file():
            continue
        try:
            _load_tt_font(FONT_REGULAR, regular_path)
            if bold_path and bold_path.is_file():
                _load_tt_font(FONT_BOLD, bold_path)
            else:
                _load_tt_font(FONT_BOLD, regular_path)
            pdfmetrics.registerFontFamily(
                FONT_REGULAR,
                normal=FONT_REGULAR,
                bold=FONT_BOLD,
                italic=FONT_REGULAR,
                boldItalic=FONT_BOLD,
            )
            return FONT_REGULAR, FONT_BOLD
        except Exception:
            continue
    raise RuntimeError(
        "找不到可用的繁體中文字型。請確認 Windows 已安裝微軟正黑體 (msjh.ttc)。"
    )


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _inline_md(text: str, font: str = FONT_REGULAR, font_b: str = FONT_BOLD) -> str:
    t = _escape(text.strip())
    t = re.sub(r"\*\*(.+?)\*\*", rf"<font name='{font_b}'>\1</font>", t)
    t = re.sub(r"`([^`]+)`", rf"<font name='{font}' size='9'>\1</font>", t)
    return t


def _parse_table_rows(lines: list[str]) -> list[list[str]] | None:
    if len(lines) < 2:
        return None
    if not all("|" in ln for ln in lines[:2]):
        return None
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    sep = lines[1].strip()
    if not re.match(r"^[\|\s:\-]+$", sep):
        return None
    rows = [header]
    for ln in lines[2:]:
        if "|" not in ln:
            break
        rows.append([c.strip() for c in ln.strip("|").split("|")])
    return rows


def _build_styles(font: str, font_b: str):
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName=font_b,
            fontSize=20,
            leading=26,
            textColor=colors.HexColor("#0284c7"),
            spaceAfter=12,
        ),
        "meta": ParagraphStyle(
            "Meta",
            parent=base["Normal"],
            fontName=font,
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#64748b"),
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName=font_b,
            fontSize=16,
            leading=22,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=14,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName=font_b,
            fontSize=13,
            leading=18,
            textColor=colors.HexColor("#1e40af"),
            spaceBefore=10,
            spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=base["Heading3"],
            fontName=font_b,
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#334155"),
            spaceBefore=8,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontName=font,
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#1e293b"),
            spaceAfter=6,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName=font,
            fontSize=8,
            leading=12,
            textColor=colors.HexColor("#334155"),
            backColor=colors.HexColor("#f8fafc"),
            leftIndent=8,
            rightIndent=8,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "li": ParagraphStyle(
            "Li",
            parent=base["Normal"],
            fontName=font,
            fontSize=10,
            leading=14,
            leftIndent=14,
            bulletIndent=6,
            spaceAfter=3,
        ),
        "th": ParagraphStyle(
            "Th",
            parent=base["Normal"],
            fontName=font_b,
            fontSize=9,
            leading=12,
            textColor=colors.white,
        ),
        "td": ParagraphStyle(
            "Td",
            parent=base["Normal"],
            fontName=font,
            fontSize=9,
            leading=12,
        ),
    }


def _table_flow(rows: list[list[str]], styles, font: str, font_b: str) -> Table:
    data = []
    for ri, row in enumerate(rows):
        style = styles["th"] if ri == 0 else styles["td"]
        data.append([Paragraph(_inline_md(c, font, font_b), style) for c in row])
    col_count = len(rows[0])
    avail = 17 * cm
    col_w = avail / col_count
    tbl = Table(data, colWidths=[col_w] * col_count, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284c7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return tbl


def md_to_story(md_text: str, styles, font: str, font_b: str) -> list:
    story = []
    lines = md_text.splitlines()
    i = 0
    in_code = False
    code_buf: list[str] = []

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()

        if line.strip().startswith("```"):
            if in_code:
                story.append(Paragraph("<br/>".join(_escape(x) for x in code_buf), styles["code"]))
                code_buf = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        if not line.strip():
            story.append(Spacer(1, 4))
            i += 1
            continue

        if line.strip() == "---":
            story.append(Spacer(1, 6))
            story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#e2e8f0")))
            story.append(Spacer(1, 6))
            i += 1
            continue

        if line.startswith("# "):
            story.append(Paragraph(_inline_md(line[2:], font, font_b), styles["title"]))
            i += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(_inline_md(line[3:], font, font_b), styles["h1"]))
            i += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(_inline_md(line[4:], font, font_b), styles["h2"]))
            i += 1
            continue

        if "|" in line and i + 1 < len(lines) and "|" in lines[i + 1]:
            block = []
            j = i
            while j < len(lines) and "|" in lines[j]:
                block.append(lines[j])
                j += 1
            rows = _parse_table_rows(block)
            if rows:
                story.append(_table_flow(rows, styles, font, font_b))
                story.append(Spacer(1, 8))
                i = j
                continue

        if line.lstrip().startswith("- "):
            story.append(
                Paragraph(f"• {_inline_md(line.lstrip()[2:], font, font_b)}", styles["li"])
            )
            i += 1
            continue

        if line.startswith("> "):
            story.append(Paragraph(_inline_md(line[2:], font, font_b), styles["meta"]))
            i += 1
            continue

        story.append(Paragraph(_inline_md(line, font, font_b), styles["body"]))
        i += 1

    return story


def export_pdf(md_path: Path, pdf_path: Path) -> Path:
    font, font_b = _register_cjk_font()
    styles = _build_styles(font, font_b)
    md_text = md_path.read_text(encoding="utf-8")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Semi-Shield ISMS 系統細部架構",
        author="Semi-Shield ISMS",
    )

    story = []
    story.append(
        Paragraph(
            f"Semi-Shield ISMS<br/>"
            f"<font name='{font}' size='12' color='#64748b'>系統細部架構文件</font>",
            styles["title"],
        )
    )
    story.append(
        Paragraph(
            f"產出時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}　｜　來源：{md_path.name}",
            styles["meta"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0284c7")))
    story.append(Spacer(1, 10))
    story.extend(md_to_story(md_text, styles, font, font_b))

    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.HexColor("#94a3b8"))
        canvas.drawString(2 * cm, 1.2 * cm, "Semi-Shield ISMS — Architecture")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"第 {doc_.page} 頁")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return pdf_path


def main() -> int:
    parser = argparse.ArgumentParser(description="匯出 ARCHITECTURE.md 為 PDF")
    parser.add_argument("-i", "--input", type=Path, default=DEFAULT_MD, help="Markdown 來源")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_PDF, help="PDF 輸出路徑")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"找不到來源檔：{args.input}", file=sys.stderr)
        return 1
    try:
        out = export_pdf(args.input, args.output)
    except Exception as e:
        print(f"匯出失敗：{e}", file=sys.stderr)
        return 1
    print(f"✅ 已匯出 PDF：{out}")
    print(f"   大小：{out.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
