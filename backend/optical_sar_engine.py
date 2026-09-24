"""
ASTRA Optical + SAR analysis baseline.

Purpose:
- Accept an optical RGB image and a SAR/grayscale image as PIL Images.
- Resize them to a common analysis size.
- Estimate basic optical and SAR scene indicators.
- Check spatial alignment using ORB + RANSAC when possible.
- Fuse complementary indicators into a lightweight cross-modal summary.

This is intentionally a classical baseline. It does NOT claim radiometric calibration,
georeferencing, true SAR backscatter calibration, or learned remote-sensing semantics.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from PIL import Image


MAX_ANALYSIS_DIM = 1024


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def _to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def _to_gray_array(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(_to_rgb(image), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _resize_keep_aspect(image: Image.Image, max_dim: int = MAX_ANALYSIS_DIM) -> Image.Image:
    image = _to_rgb(image)
    w, h = image.size

    scale = min(1.0, max_dim / max(w, h))
    if scale >= 1.0:
        return image

    new_size = (
        max(1, int(round(w * scale))),
        max(1, int(round(h * scale))),
    )
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _safe_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if arr.size else 0.0


def _safe_std(arr: np.ndarray) -> float:
    return float(np.std(arr)) if arr.size else 0.0


def _percentile(arr: np.ndarray, p: float) -> float:
    return float(np.percentile(arr, p)) if arr.size else 0.0


def _normalize_01(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _entropy(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = float(hist.sum())
    if total <= 0:
        return 0.0

    prob = hist / total
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())


def _edge_density(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0

    edges = cv2.Canny(gray, 80, 160)
    return float(np.count_nonzero(edges) / edges.size)


def _texture_score(gray: np.ndarray) -> float:
    """
    Normalized local texture indicator based on Laplacian variance.
    This is useful as a simple SAR texture proxy, not a calibrated SAR measure.
    """
    if gray.size == 0:
        return 0.0

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Soft logarithmic compression for a stable 0..1 indicator.
    return float(np.clip(math.log1p(lap_var) / math.log1p(4000.0), 0.0, 1.0))


# ---------------------------------------------------------------------
# Optical analysis
# ---------------------------------------------------------------------

def analyze_optical(image: Image.Image) -> dict[str, Any]:
    rgb = np.asarray(_to_rgb(image), dtype=np.float32) / 255.0

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Excess Green Index: simple vegetation-related RGB proxy.
    exg = 2.0 * g - r - b
    vegetation_proxy = float(np.mean(exg > 0.08))

    # Blue-dominance proxy for water.
    blue_dominance = b - (r + g) / 2.0
    water_proxy = float(np.mean(blue_dominance > 0.08))

    # Low-saturation bright pixels are often associated with roofs/roads,
    # but this is only a generic built-up/impervious proxy.
    hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    value = hsv[:, :, 2].astype(np.float32) / 255.0

    builtup_proxy = float(np.mean((saturation < 0.28) & (value > 0.35)))

    return {
        "mean_red": round(float(np.mean(r)), 4),
        "mean_green": round(float(np.mean(g)), 4),
        "mean_blue": round(float(np.mean(b)), 4),
        "mean_brightness": round(float(np.mean(gray)) / 255.0, 4),
        "vegetation_proxy": round(float(np.clip(vegetation_proxy, 0.0, 1.0)), 4),
        "water_proxy": round(float(np.clip(water_proxy, 0.0, 1.0)), 4),
        "builtup_proxy": round(float(np.clip(builtup_proxy, 0.0, 1.0)), 4),
        "edge_density": round(_edge_density(gray), 4),
        "entropy": round(_entropy(gray), 4),
        "image_width": int(image.width),
        "image_height": int(image.height),
    }


# ---------------------------------------------------------------------
# SAR analysis
# ---------------------------------------------------------------------

def analyze_sar(image: Image.Image) -> dict[str, Any]:
    gray = _to_gray_array(image)

    mean_intensity = _safe_mean(gray) / 255.0
    std_intensity = _safe_std(gray) / 255.0
    p10 = _percentile(gray, 10.0) / 255.0
    p90 = _percentile(gray, 90.0) / 255.0

    texture = _texture_score(gray)
    edge_density = _edge_density(gray)
    entropy = _entropy(gray)

    # Lightweight relative indicators.
    # These are intentionally named "proxy" / "indicator" rather than physical SAR classes.
    bright_response = _normalize_01(mean_intensity, 0.15, 0.75)
    rough_surface = float(np.clip(0.65 * texture + 0.35 * edge_density / 0.20, 0.0, 1.0))
    smooth_surface = float(np.clip(1.0 - rough_surface, 0.0, 1.0))

    return {
        "mean_intensity": round(mean_intensity, 4),
        "std_intensity": round(std_intensity, 4),
        "p10_intensity": round(float(p10), 4),
        "p90_intensity": round(float(p90), 4),
        "texture_indicator": round(texture, 4),
        "edge_density": round(edge_density, 4),
        "entropy": round(entropy, 4),
        "bright_response_indicator": round(bright_response, 4),
        "rough_surface_indicator": round(rough_surface, 4),
        "smooth_surface_indicator": round(smooth_surface, 4),
        "image_width": int(image.width),
        "image_height": int(image.height),
    }


# ---------------------------------------------------------------------
# Alignment check
# ---------------------------------------------------------------------

def estimate_alignment(
    optical: Image.Image,
    sar: Image.Image,
) -> dict[str, Any]:
    """
    Estimate visual alignment with ORB + RANSAC.

    Important:
    - Optical and SAR have different appearance, so matching may fail even when
      the images are physically co-registered.
    - Failure here means "alignment could not be verified with this feature matcher",
      not necessarily that the images are unregistered.
    """
    opt = np.asarray(_resize_keep_aspect(optical), dtype=np.uint8)
    sar = np.asarray(_resize_keep_aspect(sar), dtype=np.uint8)

    opt_gray = cv2.cvtColor(opt, cv2.COLOR_RGB2GRAY)
    sar_gray = cv2.cvtColor(sar, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)

    kp1, des1 = orb.detectAndCompute(opt_gray, None)
    kp2, des2 = orb.detectAndCompute(sar_gray, None)

    result: dict[str, Any] = {
        "status": "insufficient_features",
        "optical_keypoints": 0 if kp1 is None else len(kp1),
        "sar_keypoints": 0 if kp2 is None else len(kp2),
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
    }

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return result

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.78 * n.distance:
            good.append(m)

    result["good_matches"] = len(good)

    if len(good) < 8:
        result["status"] = "insufficient_matches"
        return result

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    try:
        _, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            5.0,
        )
    except cv2.error:
        mask = None

    if mask is None:
        result["status"] = "homography_failed"
        return result

    inliers = int(mask.ravel().sum())
    ratio = float(inliers / max(len(good), 1))

    result["inliers"] = inliers
    result["inlier_ratio"] = round(ratio, 4)
    result["status"] = "aligned_candidate" if inliers >= 8 and ratio >= 0.20 else "weak_alignment"

    return result


# ---------------------------------------------------------------------
# Cross-modal fusion
# ---------------------------------------------------------------------

def _level(score: float) -> str:
    if score >= 0.67:
        return "high"
    if score >= 0.34:
        return "medium"
    return "low"


def build_cross_modal_findings(
    optical: dict[str, Any],
    sar: dict[str, Any],
) -> list[dict[str, Any]]:
    vegetation = 0.75 * optical["vegetation_proxy"] + 0.25 * (1.0 - sar["rough_surface_indicator"])
    water = 0.85 * optical["water_proxy"] + 0.15 * sar["smooth_surface_indicator"]
    builtup = 0.55 * optical["builtup_proxy"] + 0.45 * sar["rough_surface_indicator"]

    vegetation = float(np.clip(vegetation, 0.0, 1.0))
    water = float(np.clip(water, 0.0, 1.0))
    builtup = float(np.clip(builtup, 0.0, 1.0))

    return [
        {
            "feature": "vegetation",
            "fused_score": round(vegetation, 4),
            "level": _level(vegetation),
            "explanation": "Optical greenness proxy is combined with SAR surface-texture information.",
        },
        {
            "feature": "water",
            "fused_score": round(water, 4),
            "level": _level(water),
            "explanation": "Optical blue-dominance proxy is combined with SAR smoothness.",
        },
        {
            "feature": "built_up",
            "fused_score": round(builtup, 4),
            "level": _level(builtup),
            "explanation": "Optical low-saturation/brightness proxy is combined with SAR roughness.",
        },
    ]


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------

def analyze_optical_sar(
    optical_image: Image.Image,
    sar_image: Image.Image,
) -> dict[str, Any]:
    """
    Main API used by FastAPI.

    Assumption for this baseline:
      optical_image = first uploaded image
      sar_image    = second uploaded image
    """
    if optical_image is None or sar_image is None:
        raise ValueError("Optical + SAR analysis requires two images.")

    optical = _resize_keep_aspect(optical_image)
    sar = _resize_keep_aspect(sar_image)

    # Put both modalities on a common canvas for consistent statistics.
    target_size = (
        min(optical.width, sar.width),
        min(optical.height, sar.height),
    )

    optical = optical.resize(target_size, Image.Resampling.LANCZOS)
    sar = sar.resize(target_size, Image.Resampling.LANCZOS)

    optical_stats = analyze_optical(optical)
    sar_stats = analyze_sar(sar)
    alignment = estimate_alignment(optical, sar)
    findings = build_cross_modal_findings(optical_stats, sar_stats)

    # Conservative baseline score.
    # This is explicitly a heuristic score, NOT model accuracy/probability.
    alignment_score = float(alignment.get("inlier_ratio", 0.0))
    confidence = 0.55 + 0.30 * float(np.clip(alignment_score / 0.80, 0.0, 1.0))

    if alignment["status"] in {
        "insufficient_features",
        "insufficient_matches",
        "homography_failed",
    }:
        confidence -= 0.18

    if alignment["status"] == "weak_alignment":
        confidence -= 0.10

    confidence = float(np.clip(confidence, 0.20, 0.85))

    warnings: list[str] = []

    if alignment["status"] in {
        "weak_alignment",
        "insufficient_features",
        "insufficient_matches",
        "homography_failed",
    }:
        warnings.append(
            "Optical-SAR spatial alignment could not be strongly verified "
            "with the ORB/RANSAC feature matcher."
        )

    warnings.append(
        "The fusion score is a prototype heuristic indicator and should not "
        "be interpreted as model accuracy or a calibrated probability."
    )

    high_level = ", ".join(
        f"{item['feature']}={item['level']}" for item in findings
    )

    answer = (
        "Joint optical + SAR baseline analysis completed. "
        f"Fused indicators: {high_level}. "
        "Optical imagery contributes color/appearance cues, while SAR contributes "
        "intensity and texture cues. The result uses prototype proxy indicators."
    )

    return {
        "analysis_type": "optical_sar",
        "image_size": {
            "width": int(target_size[0]),
            "height": int(target_size[1]),
        },
        "optical": optical_stats,
        "sar": sar_stats,
        "registration": alignment,
        "cross_modal_findings": findings,
        "confidence": round(confidence, 4),
        "confidence_type": "baseline_heuristic",
        "confidence_label": "Baseline Fusion Score",
        "reliability": (
            "caution"
            if alignment["status"] in {
                "weak_alignment",
                "insufficient_features",
                "insufficient_matches",
                "homography_failed",
            }
            else "normal"
        ),
        "warnings": warnings,
        "answer": answer,
        "limitations": [
            "This baseline does not perform radiometric SAR calibration.",
            "No CRS/georeferencing metadata is inferred from ordinary image files.",
            "Optical and SAR inputs are assumed to represent the same area.",
            "Proxy scores are intended for prototype evidence, not scientific measurement.",
        ],
    }
