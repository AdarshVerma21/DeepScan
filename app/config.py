from __future__ import annotations

import os
import multiprocessing
import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
REPORT_DIR = ROOT / "reports"
REPORT_RETENTION_HOURS = float(os.getenv("DEEPSCAN_REPORT_RETENTION_HOURS", "24"))
MAX_MEMORY_CACHE_ENTRIES = int(os.getenv("DEEPSCAN_MAX_MEMORY_CACHE_ENTRIES", "64"))
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
DISABLE_MODELS = os.getenv("DEEPSCAN_DISABLE_MODELS", "0").lower() in {"1", "true", "yes"}
ALLOWED_ORIGINS = tuple(
    origin.strip()
    for origin in os.getenv(
        "DEEPSCAN_ALLOWED_ORIGINS",
        "http://localhost:7860,http://127.0.0.1:7860",
    ).split(",")
    if origin.strip()
)
RATE_LIMIT_REQUESTS = int(os.getenv("DEEPSCAN_RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("DEEPSCAN_RATE_LIMIT_WINDOW_SECONDS", "60"))
MAX_VIDEO_SECONDS = 60
VIDEO_MAX_FRAMES = int(os.getenv("DEEPSCAN_VIDEO_MAX_FRAMES", "16" if DISABLE_MODELS else "60"))
VIDEO_FRAME_MAX_PX = int(os.getenv("DEEPSCAN_VIDEO_FRAME_MAX_PX", "640" if DISABLE_MODELS else "1536"))
VIDEO_BATCH_SIZE = int(os.getenv("DEEPSCAN_VIDEO_BATCH_SIZE", "4"))
VIDEO_EARLY_STOP_MIN_FRAMES = int(os.getenv("DEEPSCAN_VIDEO_EARLY_STOP_MIN_FRAMES", "8"))
VIDEO_EARLY_STOP_HIGH = float(os.getenv("DEEPSCAN_VIDEO_EARLY_STOP_HIGH", "0.95"))
VIDEO_EARLY_STOP_LOW = float(os.getenv("DEEPSCAN_VIDEO_EARLY_STOP_LOW", "0.05"))
MAX_CONCURRENT_ANALYSES = int(os.getenv("DEEPSCAN_MAX_CONCURRENT_ANALYSES", "1"))
TORCH_NUM_THREADS = int(os.getenv("DEEPSCAN_TORCH_NUM_THREADS", str(max(1, min(4, multiprocessing.cpu_count())))))
ENABLE_DYNAMIC_INT8 = os.getenv("DEEPSCAN_ENABLE_DYNAMIC_INT8", "0").lower() in {"1", "true", "yes"}
MODEL_LIMIT = max(0, int(os.getenv("DEEPSCAN_MODEL_LIMIT", "0")))
MODEL_SELECTION = tuple(
    model_id.strip()
    for model_id in os.getenv("DEEPSCAN_MODEL_SELECTION", "").split(",")
    if model_id.strip()
)
RESPONSE_IMAGE_MAX_PX = 800
RESPONSE_IMAGE_QUALITY = 78
DISK_CACHE_ENABLED = os.getenv("DEEPSCAN_DISK_CACHE", "0").lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class DetectorConfig:
    model_id: str
    weight: float


MODEL_CONFIGS = [
    DetectorConfig("dima806/deepfake_vs_real_image_detection", 1.0),
    DetectorConfig("prithivMLmods/Deep-Fake-Detector-Model", 1.0),
    DetectorConfig("umm-maybe/AI-image-detector", 0.8),
]
if MODEL_SELECTION:
    MODEL_CONFIGS = [config for config in MODEL_CONFIGS if config.model_id in MODEL_SELECTION]
if MODEL_LIMIT:
    MODEL_CONFIGS = MODEL_CONFIGS[:MODEL_LIMIT]

SIGNAL_WEIGHTS = {
    "ensemble": 0.55,
    "ela": 0.10,
    "fft": 0.10,
    "noise": 0.08,
    "metadata": 0.07,
    "boundary": 0.10,
}
TTA_ENABLED = os.getenv("DEEPSCAN_TTA_ENABLED", "1").lower() in {"1", "true", "yes"}
TTA_MARGINS = (0.12, 0.20, 0.30)
TTA_INCLUDE_FLIP = True
TTA_INSTABILITY_PENALTY = 0.15
LOW_QUALITY_SIGNAL_MULTIPLIER = 0.35
LOW_RESOLUTION_MIN_SIDE = 256
HEAVY_COMPRESSION_JPEG_QUALITY = 55
CALIBRATION_PATH = ROOT / "eval" / "calibration.json"
TUNING_PATH = ROOT / "eval" / "tuning.json"
USE_TUNING = os.getenv("DEEPSCAN_USE_TUNING", "0").lower() in {"1", "true", "yes"}
VIDEO_TEMPORAL_WEIGHT = 0.25
REAL_THRESHOLD = 30.0
DEEPFAKE_THRESHOLD = 72.0
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "30"))
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_TIMEOUT_SECONDS = float(os.getenv("ANTHROPIC_TIMEOUT_SECONDS", "30"))
ANTHROPIC_DAILY_CALL_LIMIT = int(os.getenv("ANTHROPIC_DAILY_CALL_LIMIT", "100"))
ANTHROPIC_RATE_LIMIT_REQUESTS = int(os.getenv("ANTHROPIC_RATE_LIMIT_REQUESTS", "5"))
ANTHROPIC_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("ANTHROPIC_RATE_LIMIT_WINDOW_SECONDS", "60"))

if USE_TUNING and TUNING_PATH.exists():
    try:
        tuning = json.loads(TUNING_PATH.read_text(encoding="utf-8"))
        if isinstance(tuning.get("signal_weights"), dict):
            SIGNAL_WEIGHTS = {
                name: float(tuning["signal_weights"].get(name, weight))
                for name, weight in SIGNAL_WEIGHTS.items()
            }
        REAL_THRESHOLD = float(tuning.get("real_threshold", REAL_THRESHOLD))
        DEEPFAKE_THRESHOLD = float(tuning.get("deepfake_threshold", DEEPFAKE_THRESHOLD))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RuntimeError(f"Invalid tuning file {TUNING_PATH}: {exc}") from exc
