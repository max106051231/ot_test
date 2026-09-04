"""
依 docs/ppt_assets 參考截圖，為 Semi-Shield 錄影畫面偵測場景並燒錄中文字幕。
不需 ffmpeg：OpenCV 讀寫 + Pillow 繪製中文。

用法：
  python scripts/burn_video_subtitles.py
  python scripts/burn_video_subtitles.py --video "C:\\path\\to\\capture.mp4"
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "ppt_assets"
DEFAULT_VIDEO = (
    Path(r"C:\Users\Max\Videos\Captures")
    / "Semi-Shield ISMS 整合平台 和其他 1 個頁面 - 個人 - Microsoft\u200b Edge 2026-09-02 16-59-02.mp4"
)

# 參考截圖 → 字幕（對應 generate_competition_ppt.py 說明）
SCENES: list[tuple[str, list[str]]] = [
    (
        "01_monitor_dashboard.png",
        [
            "Semi-Shield ISMS · ISO 27001 監控戰情室",
            "六控制項 KPI · 不符合統計 · 日誌串流",
            "Cisco syslog 自動對映 Annex A 控制項",
        ],
    ),
    (
        "02_model_selector.png",
        [
            "地端 LLM 模型熱切換",
            "gemma_2b_ot · qwen · phi 等微調模型可選",
        ],
    ),
    (
        "03_split_monitor_chat.png",
        [
            "整合平台：監控戰情 × AI Agent 並排檢視",
            "左：ISO 27001 監控戰情 ｜ 右：Semi-Shield Cyber Agent",
            "RAG ON · 離線就緒 · 一站式合規作業",
        ],
    ),
    (
        "04_ai_diagnosis_modal.png",
        [
            "A.8.19 地端 LLM 智慧合規診斷",
            "事件摘要 · 不合規判定 · 修補建議",
            "No Evidence, No Compliance Claim",
        ],
    ),
    (
        "05_chat_charts.png",
        [
            "AI Agent：OT 控制項事件量圖表",
            "長條圖 + 圓餅圖 · 依匯入 syslog 即時生成",
        ],
    ),
    (
        "06_platform_charts.png",
        [
            "合規控制項視覺化圖表",
            "gemma_2b_ot 生成 · 可匯出 PDF/TXT",
        ],
    ),
]

INTRO_LINES = [
    "Semi-Shield ISMS 整合平台",
    "半導體供應鏈資訊安全管理 · 地端部署 demo",
]
OUTRO_LINES = [
    "Demo：localhost:2000/platform",
    "No Evidence, No Compliance Claim",
]


@dataclass
class SubtitleCue:
    start: float
    end: float
    lines: list[str]


def manual_cues() -> list[SubtitleCue]:
    """依錄影逐格取樣校正的手動時間軸（對應 docs/ppt_assets 六張截圖說明）。"""
    return [
        SubtitleCue(0.8, 9.0, INTRO_LINES),
        SubtitleCue(
            9.0,
            34.0,
            [
                "ISO 27001 監控戰情：六控制項 KPI + 不符合統計 + 日誌串流",
                "55,966 筆 syslog 自動解析 · ot/ 目錄匯入",
            ],
        ),
        SubtitleCue(
            34.0,
            52.0,
            [
                "地端 LLM 模型熱切換",
                "微調前 / 微調後模型可選（gemma_2b_ot · Llama 3.2 3B 等）",
            ],
        ),
        SubtitleCue(
            52.0,
            58.0,
            [
                "整合平台：監控戰情 × AI Agent 並排檢視",
                "左：ISO 27001 監控戰情 ｜ 右：Semi-Shield Cyber Agent",
            ],
        ),
        SubtitleCue(
            58.0,
            96.0,
            [
                "生成稽核計畫 · 匯出稽核報告 PDF",
                "No Evidence, No Compliance Claim",
            ],
        ),
        SubtitleCue(
            96.0,
            128.0,
            [
                "六控制項 KPI：A.5.15 / A.8.19 / A.8.7 等 SYSLOG 計數",
                "優先處理事件 · 合規日誌追蹤 · 不符合統計圖",
            ],
        ),
        SubtitleCue(
            128.0,
            168.0,
            [
                "A.8.19 地端 LLM 智慧合規診斷報告",
                "事件摘要 · 不合規判定 · 修補建議",
            ],
        ),
        SubtitleCue(
            168.0,
            188.0,
            [
                "AI Agent：OT 控制項事件量圖表",
                "Semi-Shield Cyber Agent · RAG ON(681)",
            ],
        ),
        SubtitleCue(
            188.0,
            198.0,
            [
                "合規控制項視覺化 · evidence 與計數摘要",
                "gemma_2b_ot 生成 · 可匯出 PDF/TXT",
            ],
        ),
        SubtitleCue(
            198.0,
            212.0,
            [
                "Guardrails 護欄攔截越獄提示",
                "微調模型判定 unsafe · 僅限 ISO/OT 合規查詢",
            ],
        ),
        SubtitleCue(
            212.0,
            238.0,
            [
                "合規架構 · 量化 KPI 與 Annex A 覆蓋率 76%",
                "執行地端合規管線 · AI 自動審核",
            ],
        ),
        SubtitleCue(
            238.0,
            272.0,
            [
                "五層架構拓撲 · Guardrails + Human Review",
                "Collector → Auditor → Reviewer → Reporter",
            ],
        ),
        SubtitleCue(
            272.0,
            292.0,
            [
                "資安內部稽核計畫 PDF",
                "ISO 27001 Annex A · OT syslog 證據來源",
            ],
        ),
        SubtitleCue(
            292.0,
            322.0,
            [
                "資訊安全內部稽核報告",
                "KPI 不通過項 · CAPA 改善建議",
            ],
        ),
        SubtitleCue(322.0, 325.0, OUTRO_LINES),
    ]


def _load_gray(path: Path, size: tuple[int, int] = (480, 270)) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _frame_score(frame_bgr: np.ndarray, ref_gray_area: np.ndarray) -> float:
    """縮圖後比對色塊分布 + 邊緣，回傳 0~1 相似度。"""
    h, w = ref_gray_area.shape[:2]
    small = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_gray_area, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray, ref_gray)
    pixel_sim = 1.0 - float(np.mean(diff)) / 255.0

    edges_a = cv2.Canny(gray, 50, 150)
    edges_b = cv2.Canny(ref_gray, 50, 150)
    edge_diff = cv2.absdiff(edges_a, edges_b)
    edge_sim = 1.0 - float(np.mean(edge_diff)) / 255.0

    return 0.55 * pixel_sim + 0.45 * edge_sim


def detect_scene_timeline(
    video_path: Path,
    sample_every_sec: float = 1.0,
) -> tuple[float, list[tuple[str, float, float, float]]]:
    """回傳 (duration, [(asset_name, start, end, peak_score), ...])"""
    refs = {name: _load_gray(ASSETS / name) for name, _ in SCENES}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    step = max(1, int(fps * sample_every_sec))

    scores: dict[str, list[tuple[float, float]]] = {k: [] for k in refs}

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            t = frame_idx / fps
            for name, ref in refs.items():
                scores[name].append((t, _frame_score(frame, ref)))
        frame_idx += 1

    cap.release()

    segments: list[tuple[str, float, float, float]] = []
    for name, _ in SCENES:
        series = scores[name]
        if not series:
            continue
        peak_t, peak_s = max(series, key=lambda x: x[1])
        # 門檻：峰值一半以上視為同場景
        thr = max(0.42, peak_s * 0.72)
        active = [t for t, s in series if s >= thr]
        if not active:
            continue
        start, end = min(active), max(active)
        # 前後各留一點緩衝
        pad = sample_every_sec * 1.2
        start = max(0.0, start - pad)
        end = min(duration, end + pad + sample_every_sec)
        segments.append((name, start, end, peak_s))

    segments.sort(key=lambda x: x[1])
    return duration, segments


def build_cues(duration: float, segments: list[tuple[str, float, float, float]]) -> list[SubtitleCue]:
    cues: list[SubtitleCue] = []
    scene_lines = {name: lines for name, lines in SCENES}

    intro_end = segments[0][1] if segments else min(8.0, duration * 0.05)
    if intro_end > 1.0:
        cues.append(SubtitleCue(0.5, intro_end, INTRO_LINES))

    for name, start, end, _peak in segments:
        lines = scene_lines.get(name, [])
        if not lines:
            continue
        cues.append(SubtitleCue(start + 0.3, end - 0.2, lines))

    if duration - (segments[-1][2] if segments else 0) > 3:
        outro_start = max(segments[-1][2] + 0.5, duration - 6.0) if segments else duration - 6.0
        cues.append(SubtitleCue(outro_start, duration - 0.5, OUTRO_LINES))

    # 合併重疊、依時間排序
    cues.sort(key=lambda c: c.start)
    merged: list[SubtitleCue] = []
    for cue in cues:
        if merged and cue.start < merged[-1].end - 0.5:
            prev = merged[-1]
            merged[-1] = SubtitleCue(
                prev.start,
                max(prev.end, cue.end),
                prev.lines if len(prev.lines) >= len(cue.lines) else cue.lines,
            )
        else:
            merged.append(cue)
    return merged


def cues_to_srt(cues: list[SubtitleCue]) -> str:
    def fmt(sec: float) -> str:
        ms = int(round(sec * 1000))
        h, rem = divmod(ms, 3_600_000)
        m, rem = divmod(rem, 60_000)
        s, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    blocks = []
    for i, cue in enumerate(cues, 1):
        text = "\n".join(cue.lines)
        blocks.append(f"{i}\n{fmt(cue.start)} --> {fmt(cue.end)}\n{text}\n")
    return "\n".join(blocks)


def cues_to_ass(cues: list[SubtitleCue], play_res: tuple[int, int] = (1920, 1040)) -> str:
    def fmt(sec: float) -> str:
        cs = int(round(sec * 100))
        h, rem = divmod(cs, 360_000)
        m, rem = divmod(rem, 6_000)
        s, cs = divmod(rem, 100)
        return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"

    w, h = play_res
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Microsoft JhengHei,34,&H00F7EEE8,&H000000FF,&H00141008,&HCC101008,0,0,0,0,100,100,0,0,3,2,0,2,48,48,52,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for cue in cues:
        text = "\\N".join(cue.lines)
        events.append(
            f"Dialogue: 0,{fmt(cue.start)},{fmt(cue.end)},Default,,0,0,0,,{text}"
        )
    return header + "\n".join(events) + "\n"


def _ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def burn_with_ffmpeg(video_path: Path, output_path: Path, ass_path: Path) -> None:
    ffmpeg = _ffmpeg_exe()
    if not ffmpeg:
        raise RuntimeError("找不到 ffmpeg（可 pip install imageio-ffmpeg）")

    ass_esc = str(ass_path.resolve()).replace("\\", "/").replace(":", "\\:")
    vf = f"ass='{ass_esc}'"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-c:a",
        "copy",
        str(output_path),
    ]
    print("執行 ffmpeg 燒錄字幕（保留音訊）...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-4000:] or "ffmpeg 失敗")
    print(f"已輸出：{output_path}")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\msjh.ttc",
        r"C:\Windows\Fonts\msjhbd.ttc",
        r"C:\Windows\Fonts\mingliu.ttc",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_subtitle(frame_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    if not lines:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    font_size = max(22, w // 52)
    font = _font(font_size)
    line_h = font_size + 10
    pad_x, pad_y = 18, 12
    max_text_w = w - 80

    draw_probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    wrapped: list[str] = []
    for line in lines:
        words = list(line)
        chunk = ""
        for ch in words:
            test = chunk + ch
            tw = draw_probe.textlength(test, font=font)
            if tw > max_text_w and chunk:
                wrapped.append(chunk)
                chunk = ch
            else:
                chunk = test
        if chunk:
            wrapped.append(chunk)

    box_h = pad_y * 2 + line_h * len(wrapped)
    box_w = min(w - 40, int(max(draw_probe.textlength(t, font=font) for t in wrapped) + pad_x * 2))
    x0 = (w - box_w) // 2
    y0 = h - box_h - max(36, h // 14)

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(
        (x0, y0, x0 + box_w, y0 + box_h),
        radius=10,
        fill=(8, 16, 28, 210),
        outline=(78, 200, 240, 255),
        width=2,
    )
    y = y0 + pad_y
    for line in wrapped:
        tw = draw.textlength(line, font=font)
        tx = x0 + (box_w - tw) // 2
        draw.text((tx, y), line, font=font, fill=(232, 238, 247, 255))
        y += line_h

    base = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    out = Image.alpha_composite(base, overlay).convert("RGB")
    return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)


def active_lines(cues: list[SubtitleCue], t: float) -> list[str]:
    for cue in cues:
        if cue.start <= t <= cue.end:
            return cue.lines
    return []


def burn_subtitles(
    video_path: Path,
    output_path: Path,
    cues: list[SubtitleCue],
    progress_every: int = 300,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        lines = active_lines(cues, t)
        if lines:
            frame = draw_subtitle(frame, lines)
        writer.write(frame)
        idx += 1
        if idx % progress_every == 0:
            print(f"  燒錄進度 {idx} 幀 ({t:.1f}s / {idx/fps:.0f}s)", flush=True)

    cap.release()
    writer.release()
    print(f"已輸出：{output_path}")


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(encoding="utf-8")
            except Exception:
                pass


def main() -> int:
    _configure_stdio()
    parser = argparse.ArgumentParser(description="Semi-Shield 錄影字幕燒錄")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--out", type=Path, default=None, help="輸出 mp4；預設與原檔同目錄 _subtitled.mp4")
    parser.add_argument("--srt-only", action="store_true", help="只產生 SRT/ASS，不燒錄")
    parser.add_argument("--auto-detect", action="store_true", help="自動場景偵測（預設用手動時間軸）")
    parser.add_argument("--opencv", action="store_true", help="用 OpenCV 燒錄（無音訊；預設 ffmpeg）")
    parser.add_argument("--sample-sec", type=float, default=1.0)
    args = parser.parse_args()

    video = args.video.expanduser()
    if not video.is_file():
        print(f"找不到影片：{video}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(video))
    duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    cap.release()

    if args.auto_detect:
        print(f"自動分析場景：{video.name}")
        duration, segments = detect_scene_timeline(video, sample_every_sec=args.sample_sec)
        print(f"影片長度 {duration:.1f}s，偵測到 {len(segments)} 個場景")
        for name, start, end, peak in segments:
            print(f"  {name}: {start:.1f}s - {end:.1f}s (score={peak:.3f})")
        cues = build_cues(duration, segments)
    else:
        print(f"使用手動字幕時間軸：{video.name}（{duration:.1f}s）")
        cues = manual_cues()

    srt_path = ROOT / "docs" / "demo_subtitles.srt"
    ass_path = ROOT / "docs" / "demo_subtitles.ass"
    srt_path.write_text(cues_to_srt(cues), encoding="utf-8")
    ass_path.write_text(cues_to_ass(cues), encoding="utf-8-sig")
    print(f"字幕檔：{srt_path}")
    print(f"ASS 檔：{ass_path}")

    meta = {
        "video": str(video),
        "duration_sec": duration,
        "mode": "auto_detect" if args.auto_detect else "manual",
        "cues": [{"start": c.start, "end": c.end, "lines": c.lines} for c in cues],
    }
    meta_path = ROOT / "docs" / "demo_subtitles_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.srt_only:
        return 0

    out = args.out or video.with_name(video.stem + "_subtitled.mp4")
    print(f"開始燒錄字幕 → {out}")
    if args.opencv:
        burn_subtitles(video, out, cues)
    else:
        try:
            burn_with_ffmpeg(video, out, ass_path)
        except Exception as exc:
            print(f"ffmpeg 失敗，改用 OpenCV：{exc}", flush=True)
            burn_subtitles(video, out, cues)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
