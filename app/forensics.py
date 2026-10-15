from __future__ import annotations

import io
import logging
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)


def _result(
    score: float,
    finding: str,
    image: np.ndarray | None = None,
    *,
    available: bool = True,
) -> dict[str, Any]:
    return {
        "score": float(np.clip(score, 0, 1)),
        "finding": finding,
        "image": image,
        "available": available,
    }


def error_level_analysis(image: np.ndarray) -> dict[str, Any]:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    source = Image.fromarray(rgb)
    buffer = io.BytesIO()
    source.save(buffer, format="JPEG", quality=90)
    recompressed = np.asarray(Image.open(io.BytesIO(buffer.getvalue()))).astype(np.int16)
    diff = np.abs(rgb.astype(np.int16) - recompressed)
    gray = diff.mean(axis=2).astype(np.uint8)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    score = min(1.0, float(np.mean(gray) / 35.0))
    return _result(score, f"JPEG re-save error level is {np.mean(gray):.1f}/255.", cv2.applyColorMap(normalized, cv2.COLORMAP_JET))


def fft_analysis(image: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(gray)))
    log_spectrum = np.log1p(spectrum)
    visualization = cv2.normalize(log_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    high = np.mean(log_spectrum[int(len(log_spectrum) * 0.1): int(len(log_spectrum) * 0.9)])
    score = float(np.clip((high - 5.0) / 8.0, 0, 1))
    return _result(score, "Frequency spectrum shows " + ("elevated high-frequency energy." if score > 0.5 else "no strong periodic artifact signal."), cv2.applyColorMap(visualization, cv2.COLORMAP_MAGMA))


def noise_residual(image: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    residual = gray - cv2.GaussianBlur(gray, (0, 0), 2)
    height, width = residual.shape
    x1, y1, x2, y2 = (
        max(0, min(width, int(value)))
        for value in bbox
    )
    if x2 <= x1 or y2 <= y1:
        return _result(
            0.5,
            "Noise residual comparison was unavailable because the face region was invalid.",
            available=False,
        )
    face = residual[y1:y2, x1:x2]
    background_mask = np.ones((height, width), dtype=bool)
    background_mask[y1:y2, x1:x2] = False
    background = residual[background_mask]
    face_std = float(np.std(face)) if face.size else 0.0
    background_std = float(np.std(background)) if background.size else 0.0
    # A near-zero residual is not a measurable noise field. The old division by
    # background_std + 1e-6 saturated ordinary smooth backgrounds at 1.0.
    if face.size < 64 or background.size < 64 or max(face_std, background_std) < 2.0:
        return _result(
            0.5,
            (
                "Noise residual comparison was unreliable because one region "
                f"had insufficient variation (face={face_std:.2f}, background={background_std:.2f})."
            ),
            available=False,
        )
    contrast = abs(face_std - background_std) / (face_std + background_std + 1.0)
    return _result(
        contrast,
        (
            "Noise residual contrast between face and background is "
            f"{contrast:.2f} (face={face_std:.2f}, background={background_std:.2f})."
        ),
    )


def metadata_check(image_bytes: bytes, filename: str) -> dict[str, Any]:
    findings: list[str] = []
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = " ".join(str(value) for value in image.info.values()).lower()
        exif = image.getexif()
        software = str(exif.get(305, "")).lower()
        haystack = f"{text} {software} {filename.lower()}"
        signature = any(term in haystack for term in ("stable diffusion", "midjourney", "dall-e", "comfyui", "ai generated"))
        c2pa_ai = "c2pa" in haystack and any(term in haystack for term in ("ai generated", "generative", "synthetic"))
        if signature or c2pa_ai:
            findings.append("Definitive AI-generator signature found in metadata, text chunks, or C2PA manifest.")
        if software:
            findings.append(f"Editing/software tag present: {software}.")
        score = 1.0 if signature or c2pa_ai else 0.1
        if not exif:
            score = 0.5
            findings.append("No EXIF camera metadata is present; absence is not evidence of manipulation.")
        return _result(
            score,
            " ".join(findings) or "Camera metadata is present with no obvious generator signature.",
            available=bool(exif or signature or c2pa_ai),
        )
    except Exception as exc:
        LOGGER.warning("Metadata read failed: %s", exc)
        return _result(0.5, "Metadata could not be read.", available=False)


def face_boundary(image: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, Any]:
    x1, y1, x2, y2 = bbox
    if x1 == 0 and y1 == 0 and x2 == image.shape[1] and y2 == image.shape[0]:
        return _result(0.5, "No face boundary was available for blending analysis.", available=False)
    edges = cv2.Canny(image, 80, 160)
    border = np.concatenate([edges[y1:y2, x1:min(x1 + 3, x2)].ravel(), edges[y1:y2, max(x2 - 3, x1):x2].ravel()])
    sharpness = float(np.mean(border > 0)) if border.size else 0
    return _result(min(1, sharpness * 4), f"Face-crop boundary edge activity is {sharpness:.2f}.")


def run_forensics(image: np.ndarray, image_bytes: bytes, filename: str, bbox: tuple[int, int, int, int]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for name, function in (
        ("ela", lambda: error_level_analysis(image)),
        ("fft", lambda: fft_analysis(image)),
        ("noise", lambda: noise_residual(image, bbox)),
        ("metadata", lambda: metadata_check(image_bytes, filename)),
        ("boundary", lambda: face_boundary(image, bbox)),
    ):
        started = time.perf_counter()
        results[name] = function()
        results.setdefault("_timings", {})[name] = round((time.perf_counter() - started) * 1000, 2)
    results["_quality"] = input_quality(image, image_bytes)
    return results


def input_quality(image: np.ndarray, image_bytes: bytes) -> dict[str, Any]:
    height, width = image.shape[:2]
    low_resolution = min(height, width) < 256
    encoded_density = len(image_bytes) / max(width * height, 1) if image_bytes else 0.0
    heavy_compression = bool(image_bytes) and encoded_density < 0.08
    return {
        "low_resolution": low_resolution,
        "heavy_compression": heavy_compression,
        "screenshot_like": False,
        "finding": (
            "Low-resolution or heavily compressed input reduces forensic reliability."
            if low_resolution or heavy_compression
            else ""
        ),
    }
