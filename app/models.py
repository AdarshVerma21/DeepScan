from __future__ import annotations

import logging
import json
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import CALIBRATION_PATH, DISABLE_MODELS, ENABLE_DYNAMIC_INT8, MODEL_CONFIGS, TORCH_NUM_THREADS

LOGGER = logging.getLogger(__name__)
MODEL_LOAD_MS = 0.0
MODEL_WARMED = False


@lru_cache(maxsize=1)
def _calibration() -> dict[str, dict[str, float]]:
    if not CALIBRATION_PATH.exists():
        return {}
    try:
        payload = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        return {
            str(name): {"a": float(values["a"]), "b": float(values["b"])}
            for name, values in payload.items()
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        LOGGER.warning("Ignoring invalid calibration file %s: %s", CALIBRATION_PATH, exc)
        return {}


def _calibrate(name: str, probability: float) -> float:
    params = _calibration().get(name)
    if not params:
        return probability
    clipped = float(np.clip(probability, 1e-6, 1 - 1e-6))
    logit = np.log(clipped / (1 - clipped))
    return float(1 / (1 + np.exp(-(params["a"] * logit + params["b"]))))


@dataclass
class LoadedDetector:
    name: str
    processor: object
    model: object
    fake_indices: set[int]
    weight: float = 1.0

    def predict_batch(self, images: list[np.ndarray]) -> list[tuple[float, str]]:
        import torch

        inputs = self.processor(images=images, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)
        outputs = []
        for row in probabilities:
            raw_probability = float(sum(row[i] for i in self.fake_indices))
            fake_probability = _calibrate(self.name, raw_probability)
            label = str(self.model.config.id2label.get(int(torch.argmax(row)), "unknown"))
            outputs.append((fake_probability, label))
        return outputs

    def predict(self, image: np.ndarray) -> tuple[float, str]:
        return self.predict_batch([image])[0]


FAKE_LABEL_KEYWORDS = (
    "fake",
    "deepfake",
    "artificial",
    "ai",
    "ai-generated",
    "synthetic",
    "generated",
    "manipulated",
)
REAL_LABEL_KEYWORDS = ("real", "human", "authentic", "genuine", "natural")


def _matches_label(label: str, keywords: tuple[str, ...]) -> bool:
    normalized = re.sub(r"[_\s]+", "-", label.casefold())
    return any(
        keyword in normalized
        if "-" in keyword
        else re.search(rf"(?<![a-z]){re.escape(keyword)}(?![a-z])", normalized)
        for keyword in keywords
    )


def _fake_label_info(config: object) -> tuple[set[int], list[str]] | None:
    labels = getattr(config, "id2label", {}) or {}
    fake_labels = {
        int(index): str(label)
        for index, label in labels.items()
        if _matches_label(str(label), FAKE_LABEL_KEYWORDS)
        and not _matches_label(str(label), REAL_LABEL_KEYWORDS)
    }
    real_indices = {
        int(index)
        for index, label in labels.items()
        if _matches_label(str(label), REAL_LABEL_KEYWORDS)
    }
    if fake_labels:
        return set(fake_labels), list(fake_labels.values())
    if len(labels) == 2 and len(real_indices) == 1:
        other_index = next(int(index) for index in labels if int(index) not in real_indices)
        return {other_index}, [str(labels[str(other_index)]) if str(other_index) in labels else str(labels[other_index])]
    LOGGER.warning("Could not identify a fake label from id2label=%s", labels)
    return None


def _fake_indices(config: object) -> set[int]:
    info = _fake_label_info(config)
    return info[0] if info else set()


@lru_cache(maxsize=1)
def load_detectors() -> tuple[LoadedDetector, ...]:
    """Load configured detectors once; failed Hub models do not stop the app."""
    global MODEL_LOAD_MS
    if DISABLE_MODELS:
        LOGGER.warning("Pretrained detectors disabled by DEEPSCAN_DISABLE_MODELS")
        return ()
    started = __import__("time").perf_counter()
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError:
        LOGGER.warning("transformers is not installed; model ensemble is disabled")
        return ()
    import torch
    torch.set_num_threads(max(1, TORCH_NUM_THREADS))

    loaded: list[LoadedDetector] = []
    for configured in MODEL_CONFIGS:
        try:
            processor = AutoImageProcessor.from_pretrained(configured.model_id)
            model = AutoModelForImageClassification.from_pretrained(configured.model_id)
            label_info = _fake_label_info(model.config)
            if label_info is None:
                LOGGER.warning(
                    "Skipping detector %s because no fake label was identified",
                    configured.model_id,
                )
                continue
            if ENABLE_DYNAMIC_INT8 and not torch.cuda.is_available():
                model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
            LOGGER.info(
                "Loaded detector %s; id2label=%s; fake labels=%s",
                configured.model_id,
                model.config.id2label,
                label_info[1],
            )
            loaded.append(
                LoadedDetector(
                    configured.model_id, processor, model, label_info[0], configured.weight
                )
            )
        except Exception as exc:
            LOGGER.warning("Skipping detector %s: %s", configured.model_id, exc)
    MODEL_LOAD_MS = round((__import__("time").perf_counter() - started) * 1000, 2)
    LOGGER.info("Detector load completed in %.2f ms; %d detectors available", MODEL_LOAD_MS, len(loaded))
    return tuple(loaded)


def warmup_detectors() -> None:
    global MODEL_WARMED
    import torch

    detectors = load_detectors()
    if not detectors:
        return
    size = 224
    sample = np.zeros((size, size, 3), dtype=np.uint8)
    for detector in detectors:
        detector.predict(sample)
    MODEL_WARMED = True
    LOGGER.info("Detector warm-up completed on %s", "cuda" if torch.cuda.is_available() else "cpu")


def model_load_ms() -> float:
    load_detectors()
    return 0.0 if MODEL_WARMED else MODEL_LOAD_MS


def detector_status() -> dict[str, int | bool]:
    detectors = load_detectors()
    return {"available": len(detectors), "warmed": MODEL_WARMED}
