from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Profile:
    case_id: str
    period: int
    repeat_dy: int
    polarity: str
    background: tuple[int, int, int]
    sharpen_amount: float


PROFILES = {
    "pink-floral": Profile(
        case_id="pink-floral",
        period=1796,
        repeat_dy=-36,
        polarity="dark-on-light",
        background=(255, 255, 255),
        sharpen_amount=1.20,
    ),
    "white-floral": Profile(
        case_id="white-floral",
        period=1454,
        repeat_dy=18,
        polarity="light-on-dark",
        background=(0, 0, 0),
        sharpen_amount=0.0,
    ),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest().upper()


def read_rgb(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, rgb: np.ndarray, dpi: tuple[int, int] = (300, 300)) -> None:
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB").save(
        path,
        format="PNG",
        compress_level=6,
        dpi=dpi,
    )


def foreground_mask(rgb: np.ndarray, profile: Profile) -> np.ndarray:
    sample = rgb.astype(np.float32)
    height, width = rgb.shape[:2]
    band = max(12, min(height, width) // 48)
    border = np.concatenate(
        (
            sample[:band].reshape(-1, 3),
            sample[-band:].reshape(-1, 3),
            sample[:, :band].reshape(-1, 3),
            sample[:, -band:].reshape(-1, 3),
        ),
        axis=0,
    )
    background = (
        np.percentile(border, 20, axis=0)
        if profile.polarity == "light-on-dark"
        else np.median(border, axis=0)
    )
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    background_luma = float(
        background[0] * 0.2126 + background[1] * 0.7152 + background[2] * 0.0722
    )
    if profile.polarity == "light-on-dark":
        mask = gray > background_luma + 24.0
    else:
        color_distance = np.linalg.norm(sample - background[None, None, :], axis=2)
        chroma = np.max(sample, axis=2) - np.min(sample, axis=2)
        mask = (color_distance > 16.0) | (chroma > 15.0)
    solid_rows = mask.mean(axis=1) > 0.88
    solid_columns = mask.mean(axis=0) > 0.88
    mask[solid_rows, :] = False
    mask[:, solid_columns] = False
    return np.where(mask, 255, 0).astype(np.uint8)


def interpolate_profile(values: np.ndarray) -> np.ndarray:
    index = np.arange(len(values), dtype=np.float64)
    valid = np.isfinite(values)
    if np.count_nonzero(valid) < 16:
        raise RuntimeError("not enough foreground columns for geometry normalization")
    return np.interp(index, index[valid], values[valid])


def geometry_normalize(
    rgb: np.ndarray,
    mask: np.ndarray,
    profile: Profile,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    height, width = mask.shape
    slope = profile.repeat_dy / profile.period
    center_x = (width - 1) * 0.5
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    source_y = grid_y + slope * (grid_x - center_x)
    normalized = cv2.remap(
        rgb,
        grid_x,
        source_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=profile.background,
    )
    normalized_mask = cv2.remap(
        mask,
        grid_x,
        source_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return normalized, normalized_mask, {
        "method": "repeat-vector-centered-vertical-shear",
        "sourceToOutputSlope": round(-slope, 10),
        "correctionAngleDegrees": round(float(np.degrees(np.arctan(-slope))), 8),
        "centeredMaximumVerticalShiftPixels": round(abs(slope * center_x), 6),
        "repeatTranslationInput": [profile.period, profile.repeat_dy],
    }


def signal_image(rgb: np.ndarray, profile: Profile) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray if profile.polarity == "light-on-dark" else 255 - gray


def segment_quality(segment: np.ndarray, mask: np.ndarray, profile: Profile) -> dict[str, float]:
    signal = signal_image(segment, profile)
    selected = mask > 0
    background = ~selected
    laplacian = cv2.Laplacian(signal, cv2.CV_32F)
    sharpness = float(np.var(laplacian[selected])) if np.any(selected) else 0.0
    background_noise = float(np.std(signal[background])) if np.any(background) else 255.0
    foreground_ratio = float(np.mean(selected))
    score = np.log1p(sharpness) - 0.08 * background_noise - abs(foreground_ratio - 0.55) * 2.0
    return {
        "sharpness": round(sharpness, 6),
        "backgroundNoise": round(background_noise, 6),
        "foregroundRatio": round(foreground_ratio, 8),
        "score": round(float(score), 8),
    }


def align_segment(
    reference: np.ndarray,
    donor: np.ndarray,
    profile: Profile,
) -> tuple[np.ndarray, tuple[float, float], float]:
    height, width = reference.shape[:2]
    scale = min(1.0, 900.0 / max(height, width))
    size = (max(32, round(width * scale)), max(32, round(height * scale)))
    reference_small = cv2.resize(signal_image(reference, profile), size, interpolation=cv2.INTER_AREA)
    donor_small = cv2.resize(signal_image(donor, profile), size, interpolation=cv2.INTER_AREA)
    reference_float = reference_small.astype(np.float32)
    donor_float = donor_small.astype(np.float32)
    window = cv2.createHanningWindow((size[0], size[1]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(reference_float, donor_float, window)
    dx = float(np.clip(shift[0] / scale, -12.0, 12.0))
    dy = float(np.clip(shift[1] / scale, -12.0, 12.0))
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    aligned = cv2.warpAffine(
        donor,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return aligned, (dx, dy), float(response)


def circular_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    padded = np.concatenate((values, values, values))
    smoothed = cv2.GaussianBlur(padded.reshape(1, -1), (0, 0), sigma).reshape(-1)
    return smoothed[len(values) : len(values) * 2]


def normalize_template_band(
    template: np.ndarray,
    profile: Profile,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask = foreground_mask(template, profile)
    height, width = mask.shape
    top = np.full(width, np.nan, dtype=np.float64)
    bottom = np.full(width, np.nan, dtype=np.float64)
    for x in range(width):
        ys = np.flatnonzero(mask[:, x])
        if ys.size >= 4:
            top[x] = float(ys[0])
            bottom[x] = float(ys[-1])
    top = interpolate_profile(top)
    bottom = interpolate_profile(bottom)
    sigma = max(36.0, width * 0.1)
    top_smooth = circular_smooth(top, sigma)
    bottom_smooth = circular_smooth(bottom, sigma)
    center = (top_smooth + bottom_smooth) * 0.5
    span = np.maximum(bottom_smooth - top_smooth, 1.0)
    target_center = float(np.median(center))
    target_span = float(np.median(span))
    strength = 0.62 if profile.polarity == "dark-on-light" else 0.42
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    corrected_y = center[None, :] + (grid_y - target_center) * (
        span[None, :] / max(target_span, 1.0)
    )
    source_y = grid_y * (1.0 - strength) + corrected_y.astype(np.float32) * strength
    normalized = cv2.remap(
        template,
        grid_x,
        source_y.astype(np.float32),
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=profile.background,
    )
    return normalized, {
        "method": "circular-slow-envelope-width-and-center-normalization",
        "strength": strength,
        "smoothingSigmaPixels": round(sigma, 6),
        "slowSpanCvBefore": round(float(np.std(span) / np.mean(span)), 8),
        "targetSpanPixels": round(target_span, 6),
        "targetCenterY": round(target_center, 6),
    }


def build_repeat_template(
    normalized: np.ndarray,
    normalized_mask: np.ndarray,
    profile: Profile,
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = normalized.shape[:2]
    period = profile.period
    full_starts = list(range(0, width - period + 1, period))
    if len(full_starts) < 2:
        raise RuntimeError("at least two full repeat units are required")
    segments = [normalized[:, start : start + period].copy() for start in full_starts]
    segment_masks = [normalized_mask[:, start : start + period].copy() for start in full_starts]
    qualities = [
        {"repeatIndex": index, "startX": start, **segment_quality(segment, segment_mask, profile)}
        for index, (start, segment, segment_mask) in enumerate(zip(full_starts, segments, segment_masks, strict=True))
    ]
    base_index = int(np.argmax([entry["score"] for entry in qualities]))
    base = segments[base_index]
    aligned: list[np.ndarray] = []
    alignment: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if index == base_index:
            aligned_segment = segment
            shift = (0.0, 0.0)
            response = 1.0
        else:
            aligned_segment, shift, response = align_segment(base, segment, profile)
        aligned.append(aligned_segment)
        alignment.append(
            {
                "repeatIndex": index,
                "shiftPixels": [round(shift[0], 6), round(shift[1], 6)],
                "phaseCorrelationResponse": round(response, 8),
            }
        )

    stack = np.stack(aligned, axis=0).astype(np.float32)
    low_stack = np.stack(
        [cv2.GaussianBlur(segment, (0, 0), 2.2) for segment in aligned],
        axis=0,
    ).astype(np.float32)
    median_low = np.median(low_stack, axis=0)
    base_float = base.astype(np.float32)
    base_low = cv2.GaussianBlur(base_float, (0, 0), 2.2)
    anomaly = np.mean(np.abs(base_low - median_low), axis=2)
    threshold = float(np.percentile(anomaly, 98.7))
    anomaly_mask = np.where(anomaly >= max(threshold, 4.0), 255, 0).astype(np.uint8)
    anomaly_mask = cv2.morphologyEx(
        anomaly_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    anomaly_mask = cv2.dilate(anomaly_mask, np.ones((5, 5), np.uint8), iterations=1)
    alpha = cv2.GaussianBlur(anomaly_mask.astype(np.float32) / 255.0, (0, 0), 2.0)
    base_high = base_float - base_low
    consensus_detail = np.median(stack - low_stack, axis=0)
    consensus = np.clip(median_low + consensus_detail, 0, 255)
    template = np.clip(
        base_float * (1.0 - alpha[:, :, None]) + consensus * alpha[:, :, None],
        0,
        255,
    ).astype(np.uint8)
    base_start = full_starts[base_index]
    blend_width = min(96, period // 8)
    boundary_report: dict[str, Any]
    next_start = base_start + period
    if next_start + blend_width * 2 <= width:
        continuation = normalized[:, next_start : next_start + blend_width * 2]
        template[:, :blend_width] = continuation[:, :blend_width]
        blend = np.linspace(0.0, 1.0, blend_width, dtype=np.float32)[None, :, None]
        template[:, blend_width : blend_width * 2] = np.clip(
            continuation[:, blend_width : blend_width * 2].astype(np.float32) * (1.0 - blend)
            + template[:, blend_width : blend_width * 2].astype(np.float32) * blend,
            0,
            255,
        ).astype(np.uint8)
        boundary_report = {
            "method": "next-repeat-continuation-with-same-phase-crossfade",
            "blendWidthPixels": blend_width,
            "contextStartX": next_start,
        }
    elif base_start >= blend_width * 2:
        predecessor = normalized[:, base_start - blend_width * 2 : base_start]
        blend = np.linspace(0.0, 1.0, blend_width, dtype=np.float32)[None, :, None]
        transition_start = period - blend_width * 2
        transition_end = period - blend_width
        template[:, transition_start:transition_end] = np.clip(
            template[:, transition_start:transition_end].astype(np.float32) * (1.0 - blend)
            + predecessor[:, :blend_width].astype(np.float32) * blend,
            0,
            255,
        ).astype(np.uint8)
        template[:, period - blend_width :] = predecessor[:, blend_width:]
        boundary_report = {
            "method": "previous-repeat-tail-with-same-phase-crossfade",
            "blendWidthPixels": blend_width,
            "contextStartX": base_start - blend_width * 2,
        }
    else:
        boundary_report = {"method": "unclosed-no-adjacent-context", "blendWidthPixels": 0}
    band_report = {
        "enabled": False,
        "reason": "preserve-scallop-and-mesh-topology-after-repeat-vector-normalization",
    }
    return template, {
        "method": "sharpest-repeat-base-with-multi-donor-consensus-anomaly-repair",
        "periodPixels": period,
        "fullRepeatCount": len(segments),
        "baseRepeatIndex": base_index,
        "qualities": qualities,
        "alignment": alignment,
        "consensusAnomalyThreshold": round(threshold, 8),
        "consensusRepairPixels": int(np.count_nonzero(alpha > 0.02)),
        "periodicBoundary": boundary_report,
        "bandNormalization": band_report,
    }


def tile_template(template: np.ndarray, width: int) -> np.ndarray:
    repeats = int(np.ceil(width / template.shape[1]))
    return np.tile(template, (1, repeats, 1))[:, :width].copy()


def smoothstep(value: np.ndarray, low: float, high: float) -> np.ndarray:
    normalized = np.clip((value - low) / max(high - low, 1e-6), 0.0, 1.0)
    return normalized * normalized * (3.0 - 2.0 * normalized)


def clean_background_and_style(tiled: np.ndarray, profile: Profile) -> tuple[np.ndarray, dict[str, Any]]:
    cleaned_tiled = tiled.copy()
    height, width = tiled.shape[:2]
    band_start, band_end = 0, height
    band_position_report: dict[str, Any] = {
        "method": "unchanged-full-height",
        "sourceRows": [0, height],
        "targetRows": [0, height],
        "translationPixels": 0,
    }
    if profile.polarity == "light-on-dark":
        gray = cv2.cvtColor(tiled, cv2.COLOR_RGB2GRAY).astype(np.float32)
        background_luma = float(np.percentile(gray, 10.0))
        row_density = np.mean(gray > background_luma + 24.0, axis=1).astype(np.float32)
        smoothed_density = cv2.GaussianBlur(row_density.reshape(-1, 1), (1, 121), 0).reshape(-1)
        active_rows = smoothed_density > 0.10
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for index, active in enumerate(active_rows):
            if active and start is None:
                start = index
            elif not active and start is not None:
                runs.append((start, index))
                start = None
        if start is not None:
            runs.append((start, len(active_rows)))
        if runs:
            detected_start, detected_end = max(runs, key=lambda entry: entry[1] - entry[0])
            if detected_end - detected_start >= round(height * 0.45):
                detected_start = max(0, detected_start - 18)
                detected_end = min(height, detected_end + 18)
                target_start = round(height * 0.085)
                translation = target_start - detected_start
                target_end = detected_end + translation
                isolated = tiled.copy()
                isolated[:detected_start] = profile.background
                isolated[detected_end:] = profile.background
                matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, float(translation)]], dtype=np.float32)
                cleaned_tiled = cv2.warpAffine(
                    isolated,
                    matrix,
                    (width, height),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=profile.background,
                )
                band_start, band_end = max(0, target_start), min(height, target_end)
                band_position_report = {
                    "method": "self-detected-primary-band-to-fixed-a3-display-margin",
                    "backgroundLumaP10": round(background_luma, 6),
                    "smoothedRowDensityThreshold": 0.10,
                    "sourceRows": [detected_start, detected_end],
                    "targetRows": [band_start, band_end],
                    "translationPixels": translation,
                    "targetTopMarginRatio": 0.085,
                }
    source_mask = foreground_mask(cleaned_tiled, profile) > 0
    source = cleaned_tiled.astype(np.float32)
    if profile.polarity == "dark-on-light":
        chroma = np.max(source, axis=2) - np.min(source, axis=2)
        darkness = 255.0 - np.mean(source, axis=2)
        strength = np.maximum(chroma * 1.45, darkness)
        alpha = smoothstep(strength, 3.5, 22.0)
        gains = np.array([1.15, 1.45, 1.35], dtype=np.float32)
        styled = 255.0 - (255.0 - source) * gains[None, None, :]
        background = np.full_like(source, 255.0)
        result = background * (1.0 - alpha[:, :, None]) + styled * alpha[:, :, None]
        threshold = {"foregroundStrengthLow": 3.5, "foregroundStrengthHigh": 22.0}
    else:
        mask = foreground_mask(cleaned_tiled, profile) > 0
        band_sample = source[band_start:band_end:4]
        column_background = np.percentile(band_sample, 10.0, axis=0).astype(np.float32)
        column_background = cv2.GaussianBlur(
            column_background.reshape(1, width, 3),
            (0, 0),
            sigmaX=48.0,
            borderType=cv2.BORDER_REFLECT_101,
        ).reshape(width, 3)
        background = np.median(column_background, axis=0)
        foreground_values = source[mask]
        high = (
            np.percentile(foreground_values, 99.7, axis=0)
            if foreground_values.size
            else np.array([255.0, 255.0, 255.0], dtype=np.float32)
        )
        denominator = np.maximum(high[None, :] - column_background, 24.0)
        normalized = np.clip(
            (source - column_background[None, :, :]) / denominator[None, :, :],
            0.0,
            1.0,
        )
        normalized = np.power(normalized, 0.72)
        signal = np.max(normalized, axis=2)
        alpha = smoothstep(signal, 0.005, 0.045)
        result = normalized * 255.0 * alpha[:, :, None]
        threshold = {
            "backgroundRgb": np.round(background, 4).tolist(),
            "columnBackgroundMethod": "p10-per-column-gaussian-48px",
            "foregroundP997Rgb": np.round(high, 4).tolist(),
            "signalLow": 0.005,
            "signalHigh": 0.045,
            "gamma": 0.72,
        }
    result = np.clip(result, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(result, (0, 0), 0.8)
    sharpened = np.clip(
        result.astype(np.float32)
        + profile.sharpen_amount * (result.astype(np.float32) - blur.astype(np.float32)),
        0,
        255,
    ).astype(np.uint8)
    if profile.polarity == "light-on-dark":
        sharpened[np.max(sharpened, axis=2) < 3] = 0
    else:
        sharpened[np.min(sharpened, axis=2) > 252] = 255
    return sharpened, {
        "method": "polarity-aware-background-normalization-color-density-and-unsharp",
        "targetBackgroundRgb": list(profile.background),
        "primaryBandRows": [band_start, band_end],
        "bandPosition": band_position_report,
        "sharpenAmount": profile.sharpen_amount,
        "thresholds": threshold,
    }


def seam_report(rgb: np.ndarray, period: int) -> dict[str, Any]:
    differences = np.mean(
        np.abs(rgb[:, 1:].astype(np.float32) - rgb[:, :-1].astype(np.float32)),
        axis=2,
    )
    ordinary = float(np.median(np.mean(differences, axis=0)))
    seam_values: list[float] = []
    for seam in range(period, rgb.shape[1], period):
        seam_values.append(float(np.mean(differences[:, seam - 1])))
    maximum = max(seam_values, default=0.0)
    return {
        "seamCount": len(seam_values),
        "ordinaryAdjacentColumnMeanDifference": round(ordinary, 8),
        "seamAdjacentColumnMeanDifferences": [round(value, 8) for value in seam_values],
        "maximumSeamToOrdinaryRatio": round(maximum / max(ordinary, 1e-6), 8),
    }


def make_overview(
    before: np.ndarray,
    output: np.ndarray,
    reference: np.ndarray,
    path: Path,
) -> None:
    panel_width = 1500
    panel_height = 520
    label_height = 38
    labels = ("SCAN INPUT", "AUTOMATED FULL REFINE", "MANUAL REFERENCE")
    images = (before, output, reference)
    sheet = Image.new("RGB", (panel_width, (panel_height + label_height) * 3), "#202020")
    draw = ImageDraw.Draw(sheet)
    for index, (label, rgb) in enumerate(zip(labels, images, strict=True)):
        y = index * (panel_height + label_height)
        draw.text((12, y + 12), label, fill="white")
        image = Image.fromarray(rgb, "RGB")
        image.thumbnail((panel_width, panel_height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (panel_width, panel_height), "#202020")
        canvas.paste(image, ((panel_width - image.width) // 2, (panel_height - image.height) // 2))
        sheet.paste(canvas, (0, y + label_height))
    sheet.save(path, format="PNG", compress_level=6)


def load_analysis_module(repo_root: Path) -> Any:
    source = repo_root / "scripts" / "analyze-refine-cases.py"
    spec = importlib.util.spec_from_file_location("analyze_refine_cases", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load analysis helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def optional_absolute_error(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return round(abs(float(left) - float(right)), 8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a timed full-image lace refinement benchmark.")
    parser.add_argument("case_id", choices=tuple(PROFILES))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite full-image benchmark evidence")
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = PROFILES[args.case_id]
    before_path = repo_root / "validation" / "refine" / "cases" / args.case_id / "before.jpg"
    reference_path = repo_root / "validation" / "refine" / "cases" / args.case_id / "after.jpg"
    started = time.perf_counter()
    timings: dict[str, float] = {}

    step = time.perf_counter()
    before = read_rgb(before_path)
    reference = read_rgb(reference_path)
    timings["decode"] = time.perf_counter() - step

    step = time.perf_counter()
    mask = foreground_mask(before, profile)
    normalized, normalized_mask, geometry = geometry_normalize(before, mask, profile)
    timings["geometryNormalization"] = time.perf_counter() - step

    step = time.perf_counter()
    template, repeat_report = build_repeat_template(normalized, normalized_mask, profile)
    tiled = tile_template(template, before.shape[1])
    timings["repeatConsensusAndReconstruction"] = time.perf_counter() - step

    step = time.perf_counter()
    output, style_report = clean_background_and_style(tiled, profile)
    timings["backgroundColorAndSharpen"] = time.perf_counter() - step

    step = time.perf_counter()
    save_rgb(output_dir / "full-refined.png", output)
    make_overview(before, output, reference, output_dir / "overview-comparison.png")
    timings["encodeAndQaAssets"] = time.perf_counter() - step
    timings["total"] = time.perf_counter() - started

    step = time.perf_counter()
    analysis = load_analysis_module(repo_root)
    metrics = {
        "input": analysis.analyze(before_path),
        "output": analysis.analyze(output_dir / "full-refined.png"),
        "manualReference": analysis.analyze(reference_path),
    }
    timings["postRunEvaluation"] = time.perf_counter() - step
    metrics["comparisons"] = {
        "outputForegroundMedianRgbDistanceToManual": round(
            float(
                np.linalg.norm(
                    np.asarray(metrics["output"]["foreground_median_rgb"], dtype=np.float64)
                    - np.asarray(metrics["manualReference"]["foreground_median_rgb"], dtype=np.float64)
                )
            ),
            6,
        ),
        "inputForegroundMedianRgbDistanceToManual": round(
            float(
                np.linalg.norm(
                    np.asarray(metrics["input"]["foreground_median_rgb"], dtype=np.float64)
                    - np.asarray(metrics["manualReference"]["foreground_median_rgb"], dtype=np.float64)
                )
            ),
            6,
        ),
        "outputEdgeRatioAbsoluteErrorToManual": round(
            abs(float(metrics["output"]["edge_pixel_ratio"]) - float(metrics["manualReference"]["edge_pixel_ratio"])),
            8,
        ),
        "inputEdgeRatioAbsoluteErrorToManual": round(
            abs(float(metrics["input"]["edge_pixel_ratio"]) - float(metrics["manualReference"]["edge_pixel_ratio"])),
            8,
        ),
        "outputBandSpanCvAbsoluteErrorToManual": optional_absolute_error(
            metrics["output"]["band_span_cv"], metrics["manualReference"]["band_span_cv"]
        ),
        "inputBandSpanCvAbsoluteErrorToManual": optional_absolute_error(
            metrics["input"]["band_span_cv"], metrics["manualReference"]["band_span_cv"]
        ),
    }
    metrics["periodicSeams"] = seam_report(output, profile.period)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    evidence = {
        "schemaVersion": 1,
        "runId": output_dir.name,
        "caseId": profile.case_id,
        "method": "full-frame-geometry-normalization-repeat-consensus-reconstruction-and-polarity-style",
        "source": {"file": str(before_path), "sha256": digest(before_path)},
        "manualReference": {"file": str(reference_path), "sha256": digest(reference_path), "usedForProcessing": False},
        "profile": {
            "period": profile.period,
            "repeatDy": profile.repeat_dy,
            "polarity": profile.polarity,
            "background": list(profile.background),
        },
        "geometry": geometry,
        "repeatConsensus": repeat_report,
        "style": style_report,
        "timingsSeconds": {key: round(value, 6) for key, value in timings.items()},
        "outputs": {
            "fullRefined": {"file": "full-refined.png", "sha256": digest(output_dir / "full-refined.png")},
            "overviewComparison": {"file": "overview-comparison.png", "sha256": digest(output_dir / "overview-comparison.png")},
            "metrics": {"file": "metrics.json", "sha256": digest(output_dir / "metrics.json")},
        },
        "remoteSubmission": False,
        "modelGeneration": False,
    }
    (output_dir / "execution.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
