#!/usr/bin/env python3
"""
魚眼還原前／後息肉 pixel 數量一致性實驗。

用法（單張）：
  py -3.12 scripts/polyp_fisheye_pixel_experiment.py ^
    --image path/to/frame.jpg ^
    --mask path/to/polyp_mask.png ^
    --calib config/fisheye_calib.example.json ^
    --out-dir data/polyp_fisheye_experiment

用法（批次：資料夾內同名配對，如 frame001.jpg + frame001_mask.png）：
  py -3.12 scripts/polyp_fisheye_pixel_experiment.py ^
    --image-dir datasets/polyp/frames ^
    --mask-dir datasets/polyp/masks ^
    --calib config/your_calib.json ^
    --out-dir data/polyp_fisheye_experiment

輸出：summary.json、results.csv，可選預覽圖（還原前後疊 mask）。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SampleResult:
    sample_id: str
    pixels_before: int
    pixels_after: int
    diff: int
    rel_diff_pct: float
    identical: bool
    image_path: str
    mask_path: str


def _load_calib(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    K = np.array(data["K"], dtype=np.float64)
    D = np.array(data["D"], dtype=np.float64).reshape(-1, 1)
    if D.shape[0] not in (4, 5):
        raise ValueError("fisheye D 需為 4 或 5 個係數")
    if D.shape[0] == 5:
        D = D[:4]
    balance = float(data.get("balance", 0.0))
    model = (data.get("model") or "fisheye").lower()
    return {"K": K, "D": D, "balance": balance, "model": model}


def _read_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"無法讀取 mask：{path}")
    return (m > 127).astype(np.uint8)


def _count_polyp_pixels(mask_bin: np.ndarray) -> int:
    return int(np.count_nonzero(mask_bin))


def _build_fisheye_maps(
    w: int, h: int, K: np.ndarray, D: np.ndarray, balance: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = np.eye(3, dtype=np.float64)
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), R, balance=balance
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, R, new_K, (w, h), cv2.CV_16SC2
    )
    return map1, map2, new_K


def _build_pinhole_maps(
    w: int, h: int, K: np.ndarray, D: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1.0, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, None, new_K, (w, h), cv2.CV_16SC2
    )
    return map1, map2


def undistort_image_and_mask(
    image_bgr: np.ndarray,
    mask_bin: np.ndarray,
    calib: dict,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image_bgr.shape[:2]
    if mask_bin.shape[:2] != (h, w):
        raise ValueError(f"mask 尺寸 {mask_bin.shape[:2]} 與影像 {(h, w)} 不一致")

    K, D = calib["K"], calib["D"]
    model = calib["model"]

    if model == "fisheye":
        map1, map2, _ = _build_fisheye_maps(w, h, K, D, calib["balance"])
    elif model in ("pinhole", "standard"):
        D_flat = D.reshape(-1)
        if D_flat.size >= 5:
            D_use = D_flat[:5]
        else:
            D_use = np.zeros(5, dtype=np.float64)
            D_use[: D_flat.size] = D_flat
        map1, map2 = _build_pinhole_maps(w, h, K, D_use)
    else:
        raise ValueError(f"不支援的 model：{model}")

    img_out = cv2.remap(
        image_bgr, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    mask_out = cv2.remap(
        mask_bin, map1, map2, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
    )
    mask_out = (mask_out > 0).astype(np.uint8)
    return img_out, mask_out


def run_one(
    image_path: Path,
    mask_path: Path,
    calib: dict,
    *,
    tolerance: int,
    save_preview: bool,
    out_dir: Path,
    sample_id: str | None = None,
) -> SampleResult:
    sid = sample_id or image_path.stem
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"無法讀取影像：{image_path}")
    mask_bin = _read_mask(mask_path)

    pixels_before = _count_polyp_pixels(mask_bin)
    img_ud, mask_ud = undistort_image_and_mask(img, mask_bin, calib)
    pixels_after = _count_polyp_pixels(mask_ud)

    diff = pixels_after - pixels_before
    rel = (100.0 * abs(diff) / pixels_before) if pixels_before else 0.0
    identical = abs(diff) <= tolerance

    if save_preview:
        prev_dir = out_dir / "previews"
        prev_dir.mkdir(parents=True, exist_ok=True)
        overlay_before = img.copy()
        overlay_before[mask_bin > 0] = (
            0.5 * overlay_before[mask_bin > 0] + np.array([0, 0, 255]) * 0.5
        ).astype(np.uint8)
        overlay_after = img_ud.copy()
        overlay_after[mask_ud > 0] = (
            0.5 * overlay_after[mask_ud > 0] + np.array([0, 255, 0]) * 0.5
        ).astype(np.uint8)
        cv2.imwrite(str(prev_dir / f"{sid}_before.jpg"), overlay_before)
        cv2.imwrite(str(prev_dir / f"{sid}_after.jpg"), overlay_after)

    return SampleResult(
        sample_id=sid,
        pixels_before=pixels_before,
        pixels_after=pixels_after,
        diff=diff,
        rel_diff_pct=round(rel, 6),
        identical=identical,
        image_path=str(image_path.resolve()),
        mask_path=str(mask_path.resolve()),
    )


def _pair_batch(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path, str]]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)
    pairs: list[tuple[Path, Path, str]] = []
    for img_path in images:
        stem = img_path.stem
        candidates = [
            mask_dir / f"{stem}_mask.png",
            mask_dir / f"{stem}_mask.jpg",
            mask_dir / f"{stem}.png",
            mask_dir / f"{stem}.jpg",
            mask_dir / img_path.name,
        ]
        mask_path = next((p for p in candidates if p.is_file()), None)
        if mask_path is None:
            print(f"⚠️ 略過（找不到 mask）：{img_path.name}", file=sys.stderr)
            continue
        pairs.append((img_path, mask_path, stem))
    return pairs


def write_outputs(results: list[SampleResult], out_dir: Path, calib_path: Path, tolerance: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "pixels_before",
                "pixels_after",
                "diff",
                "rel_diff_pct",
                "identical",
                "image_path",
                "mask_path",
            ],
        )
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))

    n = len(results)
    n_ident = sum(1 for r in results if r.identical)
    summary = {
        "experiment": "polyp_fisheye_pixel_count",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "calib": str(calib_path.resolve()),
        "tolerance_pixels": tolerance,
        "n_samples": n,
        "n_identical_within_tolerance": n_ident,
        "all_identical": n > 0 and n_ident == n,
        "results": [asdict(r) for r in results],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="魚眼還原前後息肉 pixel 數量比較")
    ap.add_argument("--image", type=Path, help="單張影像")
    ap.add_argument("--mask", type=Path, help="單張息肉 mask（白=息肉）")
    ap.add_argument("--image-dir", type=Path, help="批次影像資料夾")
    ap.add_argument("--mask-dir", type=Path, help="批次 mask 資料夾")
    ap.add_argument(
        "--calib",
        type=Path,
        default=ROOT / "config" / "fisheye_calib.example.json",
        help="相機標定 JSON（K、D、model）",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "polyp_fisheye_experiment",
        help="結果輸出目錄",
    )
    ap.add_argument(
        "--tolerance",
        type=int,
        default=0,
        help="視為『相同』的最大 pixel 差異（插值邊界可設 1～5）",
    )
    ap.add_argument("--preview", action="store_true", help="輸出還原前後 overlay 預覽圖")
    args = ap.parse_args()

    calib_path = args.calib if args.calib.is_absolute() else ROOT / args.calib
    calib = _load_calib(calib_path)
    results: list[SampleResult] = []

    if args.image and args.mask:
        results.append(
            run_one(
                args.image if args.image.is_absolute() else ROOT / args.image,
                args.mask if args.mask.is_absolute() else ROOT / args.mask,
                calib,
                tolerance=args.tolerance,
                save_preview=args.preview,
                out_dir=args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir,
            )
        )
    elif args.image_dir and args.mask_dir:
        img_dir = args.image_dir if args.image_dir.is_absolute() else ROOT / args.image_dir
        mask_dir = args.mask_dir if args.mask_dir.is_absolute() else ROOT / args.mask_dir
        out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
        for img_p, mask_p, sid in _pair_batch(img_dir, mask_dir):
            results.append(
                run_one(
                    img_p,
                    mask_p,
                    calib,
                    tolerance=args.tolerance,
                    save_preview=args.preview,
                    out_dir=out_dir,
                    sample_id=sid,
                )
            )
    else:
        ap.error("請提供 --image + --mask，或 --image-dir + --mask-dir")

    if not results:
        print("沒有有效樣本。", file=sys.stderr)
        return 1

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    write_outputs(results, out_dir, calib_path, args.tolerance)

    for r in results:
        flag = "OK" if r.identical else "DIFF"
        print(
            f"[{flag}] {r.sample_id}: before={r.pixels_before} after={r.pixels_after} "
            f"diff={r.diff} ({r.rel_diff_pct}%)"
        )
    print(f"\n已寫入：{out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
