from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np

from .config import LOW_RESOLUTION_MIN_SIDE, TTA_ENABLED, TTA_INCLUDE_FLIP, TTA_INSTABILITY_PENALTY, TTA_MARGINS
from .face import FaceResult, detect_face, face_variant
from .forensics import run_forensics
from .models import load_detectors, model_load_ms

LOGGER = logging.getLogger(__name__)


def _score_detectors(detectors: tuple, variants: list[np.ndarray], face_crop: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, float]]:
    outputs = []
    timings: dict[str, float] = {}
    for detector in detectors:
        started = time.perf_counter()
        try:
            predictions = detector.predict_batch(variants)
            scores = [item[0] for item in predictions]
            instability = float(np.std(scores))
            face_label = predictions[0][1]
            outputs.append({
                "model": detector.name,
                "weight": detector.weight,
                "fake_probability": float(np.mean(scores)),
                "raw_fake_probability": float(np.mean(scores)),
                "instability": instability,
                "variant_count": len(scores),
                "label": face_label,
            })
        except Exception as exc:
            LOGGER.warning("Detector %s failed: %s", detector.name, exc)
        timings[f"model_inference:{detector.name}"] = round((time.perf_counter() - started) * 1000, 2)
    return outputs, timings


def _context(image: np.ndarray) -> tuple[np.ndarray, FaceResult, list[np.ndarray], dict[str, float]]:
    started = time.perf_counter()
    height, width = image.shape[:2]
    if max(height, width) > 1536:
        scale = 1536 / max(height, width)
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    timings = {"downscale": round((time.perf_counter() - started) * 1000, 2)} if max(height, width) > 1536 else {}
    face_started = time.perf_counter()
    face = detect_face(image)
    timings["face_detection"] = round((time.perf_counter() - face_started) * 1000, 2)
    variants = [image, *(face_variant(image, face, margin) for margin in TTA_MARGINS)]
    if TTA_INCLUDE_FLIP:
        variants.append(cv2.flip(image, 1))
    if not TTA_ENABLED:
        variants = [image, face.crop]
    return image, face, variants, timings


def _limitations(face: FaceResult, outputs: list[dict[str, Any]]) -> list[str]:
    instability = float(np.mean([item["instability"] for item in outputs])) if outputs else 0.5
    return ([] if face.found else ["No face was detected; the full image was analyzed."]) + (
        [f"Test-time augmentation instability is {instability:.2f}."]
        if outputs and instability > TTA_INSTABILITY_PENALTY
        else []
    ) + ([] if outputs else ["No pretrained detector was available; score is inconclusive."])


def analyze_images_batch(images: list[np.ndarray]) -> list[dict[str, Any]]:
    contexts = [_context(image) for image in images]
    detectors = load_detectors()
    flat_variants = [variant for _, _, variants, _ in contexts for variant in variants]
    offsets = []
    offset = 0
    for _, _, variants, _ in contexts:
        offsets.append((offset, offset + len(variants)))
        offset += len(variants)
    batch_outputs: list[list[dict[str, Any]]] = [[] for _ in contexts]
    batch_timings: dict[str, float] = {}
    for detector in detectors:
        started = time.perf_counter()
        try:
            predictions = detector.predict_batch(flat_variants)
            for result_index, (start, end) in enumerate(offsets):
                scores = [item[0] for item in predictions[start:end]]
                labels = [item[1] for item in predictions[start:end]]
                batch_outputs[result_index].append({
                    "model": detector.name,
                    "weight": detector.weight,
                    "fake_probability": float(np.mean(scores)),
                    "raw_fake_probability": float(np.mean(scores)),
                    "instability": float(np.std(scores)),
                    "variant_count": len(scores),
                    "label": labels[0] if labels else "unknown",
                })
        except Exception as exc:
            LOGGER.warning("Detector %s failed: %s", detector.name, exc)
        batch_timings[f"model_inference:{detector.name}"] = round((time.perf_counter() - started) * 1000, 2)
    results = []
    for index, (image, face, _, context_timings) in enumerate(contexts):
        outputs = batch_outputs[index]
        weights = [float(item["weight"]) for item in outputs]
        ensemble = float(np.average([item["fake_probability"] for item in outputs], weights=weights)) if outputs else None
        timings = {
            **context_timings,
            "model_load": model_load_ms(),
            **batch_timings,
            "total_image_pipeline": round(sum(batch_timings.values()) + sum(context_timings.values()), 2),
        }
        results.append({
            "image": image,
            "face": face,
            "per_model": outputs,
            "ensemble_score": ensemble,
            "instability": float(np.mean([item["instability"] for item in outputs])) if outputs else 0.5,
            "limitations": _limitations(face, outputs),
            "timings": timings,
        })
    return results


def analyze_image(image: np.ndarray, image_bytes: bytes = b"", filename: str = "") -> dict[str, Any]:
    started = time.perf_counter()
    result = analyze_images_batch([image])[0]
    with ThreadPoolExecutor(max_workers=1) as executor:
        forensic = executor.submit(run_forensics, result["image"], image_bytes, filename, result["face"].bbox).result()
    result["forensic"] = forensic
    result["timings"].update({f"forensics:{name}": value for name, value in forensic.get("_timings", {}).items()})
    result["timings"]["forensics_total"] = round(sum(float(value) for value in forensic.get("_timings", {}).values()), 2)
    result["timings"]["total_image_pipeline"] = round((time.perf_counter() - started) * 1000, 2)
    return result
