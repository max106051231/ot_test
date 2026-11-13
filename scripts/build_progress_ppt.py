"""Generate Semi-Shield ISMS progress + 2-week plan PPT."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "Semi-Shield_ISMS_Progress_2Week_Plan_2026-09-15.pptx"

ACCENT = RGBColor(0x1A, 0x56, 0xDB)
DARK = RGBColor(0x1E, 0x29, 0x3B)
MUTED = RGBColor(0x64, 0x74, 0x8B)


def _set_title(slide, title: str, subtitle: str = "") -> None:
    slide.shapes.title.text = title
    if subtitle and len(slide.placeholders) > 1:
        slide.placeholders[1].text = subtitle
    for p in slide.shapes.title.text_frame.paragraphs:
        p.font.size = Pt(32)
        p.font.bold = True
        p.font.color.rgb = DARK


def _add_bullets(slide, lines: list[str], *, title: str | None = None) -> None:
    if title:
        _set_title(slide, title)
        body = slide.placeholders[1].text_frame
    else:
        body = slide.shapes.add_textbox(Inches(0.6), Inches(0.5), Inches(12.2), Inches(6.5)).text_frame
    body.clear()
    for i, line in enumerate(lines):
        p = body.paragraphs[0] if i == 0 else body.add_paragraph()
        p.text = line
        p.level = 0
        p.font.size = Pt(18 if not line.startswith("  ") else 16)
        p.font.color.rgb = DARK if not line.startswith("  ") else MUTED
        if line.startswith("  "):
            p.level = 1
            p.text = line.strip()


def _add_section_slide(prs: Presentation, title: str, subtitle: str) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    _set_title(slide, title, subtitle)
    slide.placeholders[1].text_frame.paragraphs[0].font.size = Pt(20)
    slide.placeholders[1].text_frame.paragraphs[0].font.color.rgb = ACCENT


def main() -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    today = date(2026, 9, 15)

    # Title
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    _set_title(
        slide,
        "Semi-Shield ISMS",
        "OT 工控資安合規平台 — 進度整理與兩週工作規劃",
    )
    sub = slide.placeholders[1].text_frame
    sub.clear()
    for i, t in enumerate(
        [
            f"報告日期：{today.isoformat()}",
            "範圍：Gemma 4 E2B OT · Ollama 後端 · Agent 對話 · 監控上下文",
        ]
    ):
        p = sub.paragraphs[0] if i == 0 else sub.add_paragraph()
        p.text = t
        p.font.size = Pt(18)
        p.font.color.rgb = MUTED

    _add_section_slide(prs, "1. 專案總覽", "Semi-Shield Cyber Agent · No Evidence, No Compliance Claim")
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    _add_bullets(
        slide,
        [
            "平台：ISO 27001 合規、OT syslog 分析、RAG、護欄、Evidence 管線",
            "技術棧：Flask + Ollama（預設）/ 本地 Transformers · web 戰情室 + Agent Chat",
            "核心模型線：Llama 3.2 3B OT · Gemma 4 E2B（微調 / QAT 基底）",
            "部署：Windows run_ollama.bat · Port 2000 · Ollama :11434",
        ],
        title="系統定位",
    )

    _add_section_slide(prs, "2. 已完成進度（近期）", "2026-08 ~ 2026-09 交付重點")
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    _add_bullets(
        slide,
        [
            "Gemma 4 E2B OT：LoRA 150 steps / 1020 samples 完成；CPU merge 產出 ~9.7GB safetensors",
            "Ollama 匯入流程：import_models_to_ollama.py · Modelfile · 10GB 微調 vs 4.3GB 基底辨識",
            "模型切換實驗：曾還原 base_gemma4_e2b_qat；config 別名與 local_finetuned_slugs 可切換",
            "E2B 推理強化：智慧 syslog 截斷（硬體事件優先）· ctx 上限 ~12k 字 · 8192 num_ctx",
            "LLM 熱切換：keep_alive 常駐 VRAM · 本地快取不釋放 · LLM_HOT_SWITCH=1",
            "體驗優化：移除「先問你好」冷啟動探測 · Gemma4 E2B 思考模式預設關閉",
            "缺陷修復：Ollama keep_alive 字串 \"-1\" → JSON 整數 -1（HTTP 400 已解）",
            "未提交程式變更：app.py / ollama_service.py / run_ollama.*（~545 行增修）",
        ],
        title="交付與工程項",
    )

    slide = prs.slides.add_slide(prs.slide_layouts[1])
    _add_bullets(
        slide,
        [
            "train_ai/models/gemma4_e2b_ot/ — merged 權重就緒",
            "train_ai/train_llm/outputs/gemma4_e2b_ot/lora_adapter/ — LoRA 備份",
            "config/ollama_models.json — UI 白名單與別名",
            "docs/ARCHITECTURE.md · TEST_REPORT · GEMMA_COMPARE — 測試與架構文件",
        ],
        title="產物與目錄",
    )

    _add_section_slide(prs, "3. 現況與待解問題", "風險與缺口")
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    _add_bullets(
        slide,
        [
            "gemma4_e2b_ot Ollama 標籤：10GB 匯入耗時長，需確認 ollama list 為微調 hash（非 4.3GB 基底）",
            "設備異常總覽題型：微調資料偏單筆 ISO 模板，聚合戰情回答仍可能弱於預期",
            "VRAM：熱切換多模型常駐需設定 OLLAMA_MAX_LOADED_MODELS 與 GPU 容量評估",
            "品質閘：CHAT_SKIP_QUALITY_CHECK=1（edge/turbo）時需回歸測試再開 strict",
            "Git：近期 LLM 改動尚未 commit，需整理 PR / 版本標記",
        ],
        title="待辦與風險",
    )

    _add_section_slide(
        prs,
        "4. 兩週規劃 — 第 1 週",
        f"{today.isoformat()} ~ 2026-09-21",
    )
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    _add_bullets(
        slide,
        [
            "【模型】完成 gemma4_e2b_ot 匯入驗證（size ~10GB、推論 smoke test、UI 切換）",
            "【對比】base_gemma4_e2b_qat vs gemma4_e2b_ot：設備異常 / syslog / ISO 控制項 各 10 題基準",
            "【穩定】全量回歸 Ollama chat（keep_alive、熱切換、無 think 模式）",
            "【監控】確認 limit=12000 未截斷 log；戰情室 + corpus 注入路徑抽樣",
            "【工程】提交並說明 app.py / ollama_service 變更；更新 docs/OLLAMA.md 環境變數表",
        ],
        title="Week 1 工作包",
    )

    _add_section_slide(
        prs,
        "5. 兩週規劃 — 第 2 週",
        "2026-09-22 ~ 2026-09-28",
    )
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    _add_bullets(
        slide,
        [
            "【資料】增補「設備異常總覽 / 多設備 KPI」微調樣本 50~100 筆（若 Week1 對比不足）",
            "【訓練】視需要增量 LoRA（≤50 steps）或僅 merge 驗證；避免過擬合 ISO 模板",
            "【品質】啟用 CHAT_STRICT_QUALITY=1 子集測試；護欄 + 幻覺 gate 與 E2B 相容",
            "【效能】VRAM 配置指南：預載 OLLAMA_PRELOAD_ALL、max loaded models、ctx/token 建議表",
            "【交付】更新 ISO_CONVERSATION_TEST_REPORT · 內部演示腳本 · 兩週結案簡報 v2",
        ],
        title="Week 2 工作包",
    )

    slide = prs.slides.add_slide(prs.slide_layouts[1])
    _add_bullets(
        slide,
        [
            "D1：gemma4_e2b_ot 匯入成功 + 首輪問答通過",
            "D3：10 題對照報告（微調 vs 基底）",
            "D5：熱切換 + 上下文 E2E 簽核",
            "D7：Week1 結案 · 決定是否 Week2 增量訓練",
            "D10：品質閘回歸 · 文件更新",
            "D14：兩週總驗收 · 演示就緒",
        ],
        title="里程碑（建議）",
    )

    slide = prs.slides.add_slide(prs.slide_layouts[1])
    _add_bullets(
        slide,
        [
            "GPU VRAM ≥ 16GB（E2B 8192 ctx + 熱切換 2 模型）",
            "Ollama 桌面 / serve 常駐 · 磁碟 ~15GB 給 gemma4_e2b_ot",
            "測試語料：ot/*.txt · 戰情室 mock · ISO Annex A 抽題表",
            "可選：OLLAMA_GEMMA4_THINK=1 A/B 對比（預設已關）",
        ],
        title="資源與依賴",
    )

    slide = prs.slides.add_slide(prs.slide_layouts[0])
    _set_title(slide, "謝謝 · Q&A", "Semi-Shield ISMS — 持續迭代合規與 OT 可觀測性")
    slide.placeholders[1].text = "文件路徑：docs/Semi-Shield_ISMS_Progress_2Week_Plan_2026-09-15.pptx"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
