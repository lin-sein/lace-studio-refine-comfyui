from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


def load_base_module(repo_root: Path) -> Any:
    source = repo_root / "scripts" / "full-image-refine-benchmark.py"
    spec = importlib.util.spec_from_file_location("full_image_refine_benchmark", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load full-image refinement module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest().upper()


def read_rgb(path: Path) -> tuple[np.ndarray, tuple[float, float]]:
    with Image.open(path) as image:
        dpi_value = image.info.get("dpi", (72.0, 72.0))
        dpi = (float(dpi_value[0]), float(dpi_value[1]))
        return np.asarray(image.convert("RGB")), dpi


def save_rgb(path: Path, rgb: np.ndarray, dpi: tuple[float, float]) -> None:
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB").save(
        path,
        format="PNG",
        compress_level=6,
        dpi=(round(dpi[0]), round(dpi[1])),
    )


def save_mask(path: Path, alpha: np.ndarray, dpi: tuple[float, float]) -> None:
    Image.fromarray(np.clip(alpha * 255.0, 0, 255).astype(np.uint8), "L").save(
        path,
        format="PNG",
        compress_level=6,
        dpi=(round(dpi[0]), round(dpi[1])),
    )


def dominant_background(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    height = rgb.shape[0]
    band = max(3, round(height * 0.06))
    border = np.concatenate((rgb[:band], rgb[-band:]), axis=0).reshape(-1, 3).astype(np.float32)
    quantized = (border // 16).astype(np.uint8)
    colors, counts = np.unique(quantized, axis=0, return_counts=True)
    dominant = colors[int(np.argmax(counts))]
    selected = border[np.all(quantized == dominant, axis=1)]
    background = np.median(selected, axis=0)
    noise = np.linalg.norm(selected - background, axis=1)
    threshold = max(16.0, float(np.percentile(noise, 99.0)) + 8.0)
    return background, selected, threshold


def foreground_mask(rgb: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    background, samples, threshold = dominant_background(rgb)
    distance = np.linalg.norm(rgb.astype(np.float32) - background[None, None, :], axis=2)
    mask = distance > threshold
    return np.where(mask, 255, 0).astype(np.uint8), {
        "estimatedBackgroundRgb": np.round(background, 4).tolist(),
        "backgroundLuma": round(float(background @ np.array([0.2126, 0.7152, 0.0722])), 6),
        "backgroundSampleStdRgb": np.std(samples, axis=0).round(4).tolist(),
        "colorDistanceThreshold": round(threshold, 6),
    }


def runs_from_flags(flags: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(flags):
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


def detect_primary_band(rgb: np.ndarray) -> tuple[int, int, dict[str, Any]]:
    height = rgb.shape[0]
    background, _, noise_threshold = dominant_background(rgb)
    strong_threshold = max(42.0, noise_threshold * 1.5)
    distance = np.linalg.norm(rgb.astype(np.float32) - background[None, None, :], axis=2)
    density = (distance > strong_threshold).mean(axis=1).astype(np.float32)
    kernel = max(31, round(height * 0.035) | 1)
    smooth = cv2.GaussianBlur(density.reshape(-1, 1), (1, kernel), 0).reshape(-1)
    threshold = max(0.045, float(np.percentile(smooth, 55.0)) * 0.18)
    runs = runs_from_flags(smooth > threshold)
    candidates = [run for run in runs if run[1] - run[0] >= round(height * 0.15)]
    if not candidates:
        return 0, height, {
            "method": "fallback-full-height",
            "rowDensityThreshold": round(threshold, 6),
            "strongColorDistanceThreshold": round(strong_threshold, 6),
            "sourceRows": [0, height],
        }
    start, end = max(candidates, key=lambda run: float(np.sum(smooth[run[0] : run[1]])))
    expansion = max(8, round(height * 0.012))
    start = max(0, start - expansion)
    end = min(height, end + expansion)
    return start, end, {
        "method": "largest-integrated-smoothed-foreground-band",
        "rowDensityThreshold": round(threshold, 6),
        "strongColorDistanceThreshold": round(strong_threshold, 6),
        "smoothingKernelRows": kernel,
        "sourceRows": [start, end],
    }


def detect_large_repeat(rgb: np.ndarray, band: tuple[int, int]) -> dict[str, Any]:
    height, width = rgb.shape[:2]
    scale = min(1.0, 800.0 / width)
    resized = cv2.resize(
        rgb,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    y0 = max(0, round(band[0] * scale))
    y1 = min(resized.shape[0], round(band[1] * scale))
    crop = resized[y0:y1]
    crop = cv2.GaussianBlur(crop, (0, 0), sigmaX=5.0, sigmaY=3.0)
    crop = (crop - crop.mean(axis=(0, 1), keepdims=True)) / (
        crop.std(axis=(0, 1), keepdims=True) + 1e-5
    )
    scaled_width = crop.shape[1]
    scores: list[tuple[float, int]] = []
    for lag in range(round(scaled_width * 0.14), round(scaled_width * 0.49)):
        score = float(np.mean(crop[:, :-lag] * crop[:, lag:]))
        scores.append((score, lag))
    peaks: list[tuple[float, int]] = []
    for index in range(1, len(scores) - 1):
        if scores[index][0] > scores[index - 1][0] and scores[index][0] >= scores[index + 1][0]:
            peaks.append(scores[index])
    separated: list[tuple[float, int]] = []
    minimum_separation = max(4, round(scaled_width * 0.025))
    for score, lag in sorted(peaks, reverse=True):
        if all(abs(lag - selected_lag) > minimum_separation for _, selected_lag in separated):
            separated.append((score, lag))
        if len(separated) >= 8:
            break
    if not separated:
        return {"enabled": False, "confidence": 0.0, "periodPixels": None, "candidates": []}
    strongest = max(score for score, _ in separated)
    eligible = [entry for entry in separated if entry[0] >= strongest * 0.80]
    selected_score, selected_lag = max(eligible, key=lambda entry: entry[1])
    confidence = max(0.0, min(1.0, selected_score))
    period = round(selected_lag / scale)
    enabled = confidence >= 0.55 and width / period >= 2.0
    return {
        "enabled": enabled,
        "confidence": round(confidence, 8),
        "periodPixels": period,
        "fullRepeatCapacity": round(width / period, 6),
        "selectionRule": "longest-peak-within-80-percent-of-strongest-large-scale-correlation",
        "candidates": [
            {"periodPixels": round(lag / scale), "correlation": round(score, 8)}
            for score, lag in sorted(separated, reverse=True)
        ],
    }


def estimate_repeat_dy(rgb: np.ndarray, period: int) -> dict[str, Any]:
    height, width = rgb.shape[:2]
    if period * 2 > width:
        return {"dyPixels": 0, "response": 0.0, "accepted": False, "reason": "less-than-two-repeats"}
    first = rgb[:, :period]
    second = rgb[:, period : period * 2]
    scale = min(1.0, 900.0 / max(height, period))
    size = (max(32, round(period * scale)), max(32, round(height * scale)))

    def signal(segment: np.ndarray) -> np.ndarray:
        sample = cv2.resize(segment, size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY).astype(np.float32)
        low = cv2.GaussianBlur(gray, (0, 0), 2.0)
        vertical_edge = cv2.Sobel(cv2.GaussianBlur(gray, (0, 0), 1.2), cv2.CV_32F, 0, 1)
        return low + vertical_edge * 0.35

    window = cv2.createHanningWindow((size[0], size[1]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(signal(first), signal(second), window)
    dx = float(shift[0] / scale)
    dy = float(shift[1] / scale)
    accepted = float(response) >= 0.30 and abs(dx) <= period * 0.05 and abs(dy) <= height * 0.04
    return {
        "dyPixels": round(dy) if accepted else 0,
        "rawShiftPixels": [round(dx, 6), round(dy, 6)],
        "response": round(float(response), 8),
        "accepted": accepted,
        "acceptanceRule": "response>=0.30-and-horizontal-phase-error<=5-percent-period",
    }


def smoothstep(value: np.ndarray, low: float, high: float) -> np.ndarray:
    normalized = np.clip((value - low) / np.maximum(high - low, 1e-6), 0.0, 1.0)
    return normalized * normalized * (3.0 - 2.0 * normalized)


def clean_preserving_color(
    rgb: np.ndarray,
    polarity: str,
    band: tuple[int, int],
    target_background: str = "auto",
    edge_cleanup: int = 50,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if target_background not in {"auto", "white", "black"}:
        raise ValueError("target_background must be auto, white, or black")
    if not isinstance(edge_cleanup, int) or edge_cleanup < 0 or edge_cleanup > 100:
        raise ValueError("edge_cleanup must be an integer from 0 to 100")
    height, width = rgb.shape[:2]
    band_start, band_end = band
    span = band_end - band_start
    target_start = max(0, round((height - span) * 0.5))
    translation = target_start - band_start
    source = rgb.astype(np.float32)
    sample = source[band_start:band_end:4]
    percentile = 10.0 if polarity == "light-on-dark" else 90.0
    column_background = np.percentile(sample, percentile, axis=0).astype(np.float32)
    sigma = max(24.0, width * 0.02)
    column_background = cv2.GaussianBlur(
        column_background.reshape(1, width, 3),
        (0, 0),
        sigmaX=sigma,
        borderType=cv2.BORDER_REFLECT_101,
    ).reshape(width, 3)
    boundary_distance = np.linalg.norm(
        source - column_background[None, :, :],
        axis=2,
    )
    # A stricter edge gate keeps broad scanner/bed shading from being mistaken
    # for lace at the top and bottom of each column.
    boundary_threshold = 42.0 + edge_cleanup * 0.36
    strong_boundary = boundary_distance > boundary_threshold
    top = np.full(width, np.nan, dtype=np.float64)
    bottom = np.full(width, np.nan, dtype=np.float64)
    for x in range(width):
        ys = np.flatnonzero(strong_boundary[band_start:band_end, x])
        if ys.size:
            top[x] = float(band_start + ys[0])
            bottom[x] = float(band_start + ys[-1])
    index = np.arange(width, dtype=np.float64)
    valid = np.isfinite(top) & np.isfinite(bottom)
    if np.count_nonzero(valid) >= max(16, round(width * 0.15)):
        top = np.interp(index, index[valid], top[valid])
        bottom = np.interp(index, index[valid], bottom[valid])
        top = cv2.GaussianBlur(top.astype(np.float32).reshape(1, -1), (0, 0), 6.0).reshape(-1) - 4.0
        bottom = cv2.GaussianBlur(bottom.astype(np.float32).reshape(1, -1), (0, 0), 6.0).reshape(-1) + 4.0
    else:
        top = np.full(width, float(band_start), dtype=np.float32)
        bottom = np.full(width, float(band_end - 1), dtype=np.float32)
    rows = np.arange(height, dtype=np.float32)[:, None]
    top_gate = smoothstep(rows, top[None, :] - 2.0, top[None, :] + 3.0)
    bottom_gate = 1.0 - smoothstep(rows, bottom[None, :] - 3.0, bottom[None, :] + 2.0)
    boundary_gate = top_gate * bottom_gate
    if polarity == "light-on-dark":
        # Use a scalar background luminance for subtraction. Per-channel
        # subtraction can turn a warm scanner bed into a complementary green
        # fringe even when the source lace contains no green thread.
        column_background_luma = (
            column_background[:, 0] * 0.2126
            + column_background[:, 1] * 0.7152
            + column_background[:, 2] * 0.0722
        )
        denominator = np.maximum(255.0 - column_background_luma, 32.0)
        normalized = np.clip(
            (source - column_background_luma[None, :, None]) / denominator[None, :, None],
            0.0,
            1.0,
        )
        signal = np.max(normalized, axis=2)
        alpha = smoothstep(signal, 0.012, 0.075)
        styled = np.power(normalized, 0.72) * 255.0
        styled_target = "black"
    else:
        denominator = np.maximum(column_background, 32.0)
        ink = np.clip(
            (column_background[None, :, :] - source) / denominator[None, :, :],
            0.0,
            1.0,
        )
        signal = np.max(ink, axis=2)
        alpha = smoothstep(signal, 0.018, 0.105)
        styled = 255.0 - np.clip(ink * 255.0 * 1.12, 0.0, 255.0)
        styled_target = "white"
    resolved_background = styled_target if target_background == "auto" else target_background
    target = (
        np.zeros_like(source)
        if resolved_background == "black"
        else np.full_like(source, 255.0)
    )
    alpha *= boundary_gate
    result = target * (1.0 - alpha[:, :, None]) + styled * alpha[:, :, None]
    isolated = target.copy()
    isolated[band_start:band_end] = result[band_start:band_end]
    isolated_alpha = np.zeros_like(alpha)
    isolated_alpha[band_start:band_end] = alpha[band_start:band_end]
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, float(translation)]], dtype=np.float32)
    border_value = (0, 0, 0) if resolved_background == "black" else (255, 255, 255)
    positioned = cv2.warpAffine(
        isolated.astype(np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    positioned_alpha = cv2.warpAffine(
        isolated_alpha.astype(np.float32),
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    positioned_alpha = np.clip(positioned_alpha, 0.0, 1.0)
    positioned[positioned_alpha < 0.05] = np.asarray(border_value, dtype=np.uint8)
    return positioned, positioned_alpha, {
        "method": "column-flat-field-soft-alpha-color-preserving-background-replacement",
        "polarity": polarity,
        "requestedBackground": target_background,
        "resolvedBackground": resolved_background,
        "edgeCleanup": edge_cleanup,
        "columnBackgroundPercentile": percentile,
        "columnBackgroundSmoothingSigmaPixels": round(sigma, 6),
        "medianColumnBackgroundRgb": np.median(column_background, axis=0).round(4).tolist(),
        "backgroundSubtraction": "scalar-luminance" if polarity == "light-on-dark" else "per-channel-ink",
        "sourceBandRows": [band_start, band_end],
        "targetBandRows": [target_start, target_start + span],
        "verticalTranslationPixels": translation,
        "alphaSignalThresholds": [0.012, 0.075] if polarity == "light-on-dark" else [0.018, 0.105],
        "columnBoundaryGate": {
            "method": "strong-color-distance-column-envelope-with-5px-soft-gate",
            "distanceThreshold": round(boundary_threshold, 4),
            "topMedianPixels": round(float(np.median(top)), 4),
            "bottomMedianPixels": round(float(np.median(bottom)), 4),
        },
    }


def refine_array(
    rgb: np.ndarray,
    *,
    repo_root: Path,
    target_background: str = "auto",
    repeat_reconstruction: str = "auto",
    edge_cleanup: int = 50,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run the v0.1 deterministic pipeline without file-system I/O."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be an HxWx3 uint8 array")
    if repeat_reconstruction not in {"auto", "off"}:
        raise ValueError("repeat_reconstruction must be auto or off")

    base = load_base_module(repo_root)
    foreground, background_report = foreground_mask(rgb)
    band_start, band_end, band_report = detect_primary_band(rgb)
    polarity = "light-on-dark" if background_report["backgroundLuma"] < 150.0 else "dark-on-light"
    repeat_report = detect_large_repeat(rgb, (band_start, band_end))
    if repeat_reconstruction == "off":
        repeat_report = {
            **repeat_report,
            "enabled": False,
            "disabledByUser": True,
        }

    if repeat_report["enabled"]:
        period = int(repeat_report["periodPixels"])
        dy_report = estimate_repeat_dy(rgb, period)
        geometry_background = (0, 0, 0) if polarity == "light-on-dark" else (255, 255, 255)
        profile = base.Profile(
            case_id="comfyui-v0.1",
            period=period,
            repeat_dy=int(dy_report["dyPixels"]),
            polarity=polarity,
            background=geometry_background,
            sharpen_amount=0.0,
        )
        normalized, normalized_mask, geometry_report = base.geometry_normalize(rgb, foreground, profile)
        template, consensus_report = base.build_repeat_template(normalized, normalized_mask, profile)
        reconstructed = base.tile_template(template, rgb.shape[1])
        consensus_report["autoRepeatDetection"] = repeat_report
        geometry_report["autoRepeatDy"] = dy_report
    else:
        reconstructed = rgb.copy()
        geometry_report = {
            "method": "passthrough-repeat-reconstruction-disabled",
            "correctionAngleDegrees": 0.0,
            "autoRepeatDy": {"accepted": False, "dyPixels": 0},
        }
        consensus_report = {
            "method": "disabled-by-user" if repeat_reconstruction == "off" else "disabled-low-large-scale-repeat-confidence",
            "autoRepeatDetection": repeat_report,
        }

    out_start, out_end, reconstructed_band_report = detect_primary_band(reconstructed)
    output, alpha, style_report = clean_preserving_color(
        reconstructed,
        polarity,
        (out_start, out_end),
        target_background=target_background,
        edge_cleanup=edge_cleanup,
    )
    report = {
        "schemaVersion": 1,
        "engine": "lace-studio-comfyui-v0.1",
        "detection": {
            "polarity": polarity,
            "background": background_report,
            "initialBand": band_report,
            "reconstructedBand": reconstructed_band_report,
            "repeat": repeat_report,
        },
        "geometry": geometry_report,
        "repeatConsensus": consensus_report,
        "style": style_report,
        "modelGeneration": False,
    }
    return output, alpha, report


def make_overview(before: np.ndarray, output: np.ndarray, alpha: np.ndarray, path: Path) -> None:
    panel_width = 1500
    panel_height = 650
    label_height = 38
    mask_rgb = np.repeat((alpha[:, :, None] * 255.0).astype(np.uint8), 3, axis=2)
    labels = ("SCAN INPUT", "AUTOMATED GENERALIZED REFINE", "SOFT FOREGROUND MASK")
    images = (before, output, mask_rgb)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a no-reference generalized lace full-image refinement.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite generalized full-image evidence")
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.input.resolve()
    base = load_base_module(repo_root)
    started = time.perf_counter()
    timings: dict[str, float] = {}

    step = time.perf_counter()
    before, dpi = read_rgb(input_path)
    mask, background_report = foreground_mask(before)
    band_start, band_end, band_report = detect_primary_band(before)
    polarity = "light-on-dark" if background_report["backgroundLuma"] < 150.0 else "dark-on-light"
    repeat_report = detect_large_repeat(before, (band_start, band_end))
    timings["decodeAndDetection"] = time.perf_counter() - step

    geometry_report: dict[str, Any]
    consensus_report: dict[str, Any]
    step = time.perf_counter()
    if repeat_report["enabled"]:
        period = int(repeat_report["periodPixels"])
        dy_report = estimate_repeat_dy(before, period)
        target_background = (0, 0, 0) if polarity == "light-on-dark" else (255, 255, 255)
        profile = base.Profile(
            case_id=args.case_id,
            period=period,
            repeat_dy=int(dy_report["dyPixels"]),
            polarity=polarity,
            background=target_background,
            sharpen_amount=0.0,
        )
        normalized, normalized_mask, geometry_report = base.geometry_normalize(before, mask, profile)
        template, consensus_report = base.build_repeat_template(normalized, normalized_mask, profile)
        reconstructed = base.tile_template(template, before.shape[1])
        consensus_report["autoRepeatDetection"] = repeat_report
        geometry_report["autoRepeatDy"] = dy_report
    else:
        reconstructed = before.copy()
        geometry_report = {
            "method": "passthrough-low-repeat-confidence",
            "correctionAngleDegrees": 0.0,
            "autoRepeatDy": {"accepted": False, "dyPixels": 0},
        }
        consensus_report = {
            "method": "disabled-low-large-scale-repeat-confidence",
            "autoRepeatDetection": repeat_report,
        }
    timings["geometryAndRepeatReconstruction"] = time.perf_counter() - step

    step = time.perf_counter()
    out_start, out_end, reconstructed_band_report = detect_primary_band(reconstructed)
    output, alpha, style_report = clean_preserving_color(
        reconstructed,
        polarity,
        (out_start, out_end),
    )
    timings["backgroundPositionAndColor"] = time.perf_counter() - step

    step = time.perf_counter()
    output_path = output_dir / "full-refined.png"
    alpha_path = output_dir / "soft-foreground-mask.png"
    overview_path = output_dir / "overview-comparison.png"
    save_rgb(output_path, output, dpi)
    save_mask(alpha_path, alpha, dpi)
    make_overview(before, output, alpha, overview_path)
    timings["encodeAndQaAssets"] = time.perf_counter() - step
    timings["total"] = time.perf_counter() - started

    step = time.perf_counter()
    analysis_spec = importlib.util.spec_from_file_location(
        "analyze_refine_cases_generalized",
        repo_root / "scripts" / "analyze-refine-cases.py",
    )
    if analysis_spec is None or analysis_spec.loader is None:
        raise RuntimeError("cannot load analysis module")
    analysis = importlib.util.module_from_spec(analysis_spec)
    analysis_spec.loader.exec_module(analysis)
    metrics = {
        "input": analysis.analyze(input_path),
        "output": analysis.analyze(output_path),
    }
    period_value = int(repeat_report["periodPixels"]) if repeat_report["enabled"] else None
    metrics["periodicSeams"] = base.seam_report(output, period_value) if period_value else None
    target = np.zeros_like(output, dtype=np.float32) if polarity == "light-on-dark" else np.full_like(output, 255.0, dtype=np.float32)
    hole_pixels = alpha < 0.05
    residual = np.linalg.norm(output.astype(np.float32) - target, axis=2)
    metrics["backgroundResidualInLowAlphaPixels"] = {
        "pixelRatio": round(float(hole_pixels.mean()), 8),
        "meanRgbDistance": round(float(np.mean(residual[hole_pixels])), 6) if np.any(hole_pixels) else None,
        "p99RgbDistance": round(float(np.percentile(residual[hole_pixels], 99)), 6) if np.any(hole_pixels) else None,
    }
    timings["postRunEvaluation"] = time.perf_counter() - step
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    evidence = {
        "schemaVersion": 1,
        "runId": output_dir.name,
        "caseId": args.case_id,
        "input": {
            "file": str(input_path),
            "sha256": digest(input_path),
            "width": before.shape[1],
            "height": before.shape[0],
            "dpi": [round(dpi[0], 3), round(dpi[1], 3)],
        },
        "referenceImage": None,
        "detection": {
            "polarity": polarity,
            "background": background_report,
            "initialBand": band_report,
            "reconstructedBand": reconstructed_band_report,
            "repeat": repeat_report,
        },
        "geometry": geometry_report,
        "repeatConsensus": consensus_report,
        "style": style_report,
        "timingsSeconds": {key: round(value, 6) for key, value in timings.items()},
        "outputs": {
            "fullRefined": {"file": output_path.name, "sha256": digest(output_path)},
            "softForegroundMask": {"file": alpha_path.name, "sha256": digest(alpha_path)},
            "overviewComparison": {"file": overview_path.name, "sha256": digest(overview_path)},
            "metrics": {"file": "metrics.json", "sha256": digest(output_dir / "metrics.json")},
        },
        "remoteSubmission": False,
        "modelGeneration": False,
        "productionReady": False,
    }
    (output_dir / "execution.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
