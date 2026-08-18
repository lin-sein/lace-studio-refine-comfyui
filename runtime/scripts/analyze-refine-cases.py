from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def resize_for_analysis(rgb: np.ndarray, max_side: int = 1800) -> tuple[np.ndarray, float]:
    height, width = rgb.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return rgb, scale
    resized = cv2.resize(
        rgb,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def dominant_border_background(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height = rgb.shape[0]
    band = max(2, round(height * 0.06))
    border = np.concatenate((rgb[:band], rgb[-band:]), axis=0).reshape(-1, 3)
    quantized = (border // 16).astype(np.uint8)
    colors, counts = np.unique(quantized, axis=0, return_counts=True)
    dominant = colors[int(np.argmax(counts))]
    selected = border[np.all(quantized == dominant, axis=1)]
    background = np.median(selected, axis=0)
    return background, selected


def foreground_geometry(mask: np.ndarray, scale: float) -> dict[str, float | None]:
    height, width = mask.shape
    spans = np.full(width, np.nan, dtype=np.float64)
    tops = np.full(width, np.nan, dtype=np.float64)
    bottoms = np.full(width, np.nan, dtype=np.float64)

    for x in range(width):
        ys = np.flatnonzero(mask[:, x])
        if ys.size:
            tops[x] = float(ys[0])
            bottoms[x] = float(ys[-1])
            spans[x] = float(ys[-1] - ys[0] + 1)

    inset = max(1, round(width * 0.02))
    core = spans[inset:-inset]
    core = core[np.isfinite(core)]
    if core.size < 10:
        return {
            "detected_column_ratio": 0.0,
            "band_span_median_px": None,
            "band_span_p10_px": None,
            "band_span_p90_px": None,
            "band_span_cv": None,
            "top_baseline_angle_deg": None,
            "bottom_baseline_angle_deg": None,
        }

    p05, p95 = np.percentile(core, (5, 95))
    x_values = np.arange(width, dtype=np.float64)
    fit_mask = (
        np.isfinite(spans)
        & (spans >= p05)
        & (spans <= p95)
        & (x_values >= inset)
        & (x_values < width - inset)
    )

    def angle(values: np.ndarray) -> float | None:
        if np.count_nonzero(fit_mask) < 10:
            return None
        slope = float(np.polyfit(x_values[fit_mask], values[fit_mask], 1)[0])
        return round(math.degrees(math.atan(slope)), 4)

    factor = 1.0 / scale
    mean = float(np.mean(core))
    return {
        "detected_column_ratio": round(float(core.size / max(1, width - inset * 2)), 4),
        "band_span_median_px": round(float(np.median(core) * factor), 1),
        "band_span_p10_px": round(float(np.percentile(core, 10) * factor), 1),
        "band_span_p90_px": round(float(np.percentile(core, 90) * factor), 1),
        "band_span_cv": round(float(np.std(core) / mean), 4) if mean else None,
        "top_baseline_angle_deg": angle(tops),
        "bottom_baseline_angle_deg": angle(bottoms),
    }


def candidate_repeat_period(mask: np.ndarray, scale: float) -> float | None:
    profile = mask.mean(axis=0).astype(np.float64)
    profile -= cv2.GaussianBlur(profile.reshape(1, -1), (0, 0), 18).reshape(-1)
    profile -= profile.mean()
    if np.allclose(profile, 0):
        return None
    spectrum = np.fft.rfft(profile, n=profile.size * 2)
    correlation = np.fft.irfft(spectrum * np.conjugate(spectrum))[: profile.size]
    low = max(8, round(profile.size * 0.06))
    high = max(low + 1, round(profile.size * 0.42))
    window = correlation[low:high]
    if window.size == 0:
        return None
    lag = low + int(np.argmax(window))
    return round(float(lag / scale), 1)


def analyze(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        dpi = image.info.get("dpi", (None, None))
        rgb = np.asarray(image.convert("RGB"))
        original_height, original_width = rgb.shape[:2]

    sample, scale = resize_for_analysis(rgb)
    background, background_samples = dominant_border_background(sample)
    background_distances = np.linalg.norm(background_samples.astype(np.float32) - background, axis=1)
    threshold = max(24.0, float(np.percentile(background_distances, 99)) + 12.0)
    gray = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
    distance = np.linalg.norm(sample.astype(np.float32) - background, axis=2)
    background_luma = float(
        0.2126 * background[0] + 0.7152 * background[1] + 0.0722 * background[2]
    )
    if background_luma < 160:
        # White lace on a dark scan bed is better separated by signed luminance.
        # Color distance alone mistakes broad illumination shifts for lace.
        foreground = gray.astype(np.float32) > background_luma + 36.0
    else:
        foreground = distance > threshold

    # Scanner paper/bed borders can be almost solid across an entire row or
    # column. They are acquisition artifacts, not lace geometry.
    solid_rows = foreground.mean(axis=1) > 0.96
    solid_columns = foreground.mean(axis=0) > 0.96
    foreground[solid_rows, :] = False
    foreground[:, solid_columns] = False

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edges = cv2.Canny(gray, 60, 160)
    foreground_pixels = sample[foreground]
    foreground_median = (
        np.median(foreground_pixels, axis=0).round(1).tolist()
        if foreground_pixels.size
        else None
    )

    result: dict[str, object] = {
        "file": str(path.as_posix()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "width_px": original_width,
        "height_px": original_height,
        "dpi": [round(float(dpi[0]), 2), round(float(dpi[1]), 2)]
        if dpi[0] is not None and dpi[1] is not None
        else None,
        "analysis_scale": round(scale, 6),
        "estimated_background_rgb": np.round(background, 1).tolist(),
        "background_sample_channel_std": np.std(background_samples, axis=0).round(2).tolist(),
        "foreground_median_rgb": foreground_median,
        "foreground_pixel_ratio": round(float(foreground.mean()), 4),
        "near_white_pixel_ratio": round(float(np.all(sample >= 248, axis=2).mean()), 4),
        "near_black_pixel_ratio": round(float(np.all(sample <= 7, axis=2).mean()), 4),
        "edge_pixel_ratio": round(float((edges > 0).mean()), 4),
        "laplacian_variance": round(float(laplacian.var()), 2),
        "candidate_repeat_period_px": candidate_repeat_period(foreground, scale),
    }
    result.update(foreground_geometry(foreground, scale))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure geometry, background, detail, and repeat cues in lace refine cases."
    )
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()
    missing = [path for path in args.images if not path.is_file()]
    if missing:
        parser.error("missing image(s): " + ", ".join(str(path) for path in missing))
    print(json.dumps([analyze(path.resolve()) for path in args.images], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
