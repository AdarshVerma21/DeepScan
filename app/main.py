from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import tempfile
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import (
    ALLOWED_ORIGINS,
    DISK_CACHE_ENABLED,
    MAX_CONCURRENT_ANALYSES,
    MAX_MEMORY_CACHE_ENTRIES,
    MAX_UPLOAD_BYTES,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    REPORT_DIR,
    REPORT_RETENTION_HOURS,
    STATIC_DIR,
)
from .explain import heatmap_overlay
from .explain import image_to_data_url
from .forensics import run_forensics
from .fusion import fuse
from .image_pipeline import analyze_image
from .models import detector_status, warmup_detectors
from .report import save_report
from .schemas import AnalysisResponse
from .summary import generate_summary
from .video_pipeline import analyze_video

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _cleanup_artifacts()
    await asyncio.to_thread(warmup_detectors)
    yield


app = FastAPI(title="DeepScan", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
REPORTS: dict[str, dict] = {}
ANALYSIS_SEMAPHORE = asyncio.Semaphore(max(1, MAX_CONCURRENT_ANALYSES))
CACHE: dict[str, dict] = {}
REQUESTS_BY_CLIENT: dict[str, deque[float]] = defaultdict(deque)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_SIGNATURES = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"BM")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    return response


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _check_rate_limit(request: Request) -> None:
    now = time.monotonic()
    timestamps = REQUESTS_BY_CLIENT[_client_key(request)]
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    while timestamps and timestamps[0] <= cutoff:
        timestamps.popleft()
    if len(timestamps) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(429, "Too many analysis requests. Please try again later.")
    timestamps.append(now)


def _cleanup_artifacts() -> None:
    """Apply the configured privacy retention policy to derived artifacts."""
    if REPORT_RETENTION_HOURS <= 0:
        return
    cutoff = time.time() - REPORT_RETENTION_HOURS * 3600
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for path in REPORT_DIR.iterdir():
        if path.suffix not in {".html", ".json"}:
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError as exc:
            LOGGER.warning("Could not remove expired artifact %s: %s", path, exc)


def _cache_payload(cache_key: str, payload: dict) -> None:
    CACHE[cache_key] = payload
    while len(CACHE) > max(1, MAX_MEMORY_CACHE_ENTRIES):
        CACHE.pop(next(iter(CACHE)))


def _reason_chips(evidence: list[dict]) -> list[dict[str, str]]:
    chips = []
    for item in evidence:
        score = float(item.get("score", 0.5))
        direction = "fake" if score >= 0.65 else ("real" if score <= 0.35 else "neutral")
        chips.append(
            {
                "label": f"{item.get('signal', 'Signal')}: {round(score * 100)}% fake",
                "direction": direction,
                "detail": str(item.get("finding", "")),
            }
        )
    return chips[:5]


def _thumbnail(image: np.ndarray) -> str:
    height, width = image.shape[:2]
    scale = min(1.0, 900 / max(height, width))
    if scale < 1:
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return image_to_data_url(image)


CAPTIONS = {
    "face_crop": "This is the region DeepScan analyzed most closely. Look for unnatural skin texture, eyes, hair, or facial edges.",
    "ela": "Error Level Analysis (ELA) compares a JPEG with a fresh save. Bright patches mean that region was compressed differently, which can indicate editing.",
    "fft": "The frequency map shows repeating fine-detail patterns. Unusual bright bands can be a clue, but ordinary resizing and compression can also create them.",
    "heatmap": "A heatmap highlights areas that contributed more to the visual explanation. Blue looks more natural; red is the most suspicious.",
}


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "DeepScan"}


@app.get("/api/ready")
async def ready() -> dict[str, object]:
    status = detector_status()
    if not status["available"]:
        raise HTTPException(503, "No pretrained detectors are available.")
    return {"status": "ready", "service": "DeepScan", "models": status}


def _decode_image(raw: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file is not a readable image or video.")
    return image


def _validate_upload_content(raw: bytes, suffix: str) -> None:
    """Reject files whose bytes do not match the selected media family."""
    if suffix in {".jpg", ".jpeg", ".png", ".bmp"}:
        if not raw.startswith(IMAGE_SIGNATURES):
            raise ValueError("The uploaded file is not a valid image.")
        _decode_image(raw)
    elif suffix == ".webp":
        if not (raw.startswith(b"RIFF") and raw[8:12] == b"WEBP"):
            raise ValueError("The uploaded file is not a valid WebP image.")
        _decode_image(raw)
    else:
        # OpenCV performs the authoritative container/codec validation in
        # analyze_video after the bounded temporary file is created.
        if len(raw) < 12:
            raise ValueError("The uploaded file is not a valid video.")


async def _analyze(raw: bytes, filename: str, client_key: str = "unknown") -> dict:
    started = time.perf_counter()
    suffix = Path(filename).suffix.lower()
    image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    report_id = uuid.uuid4().hex
    if suffix in image_suffixes:
        image = _decode_image(raw)
        result = await asyncio.to_thread(analyze_image, image, raw, filename)
        forensic = result["forensic"]
        fusion = fuse(result["ensemble_score"], forensic, limitations=result["limitations"])
        heatmaps = [heatmap_overlay(result["face"].crop)]
        visualizations = {
            "face_crop": _thumbnail(result["face"].crop),
            "ela": image_to_data_url(forensic["ela"]["image"]) if forensic["ela"].get("image") is not None else "",
            "fft": image_to_data_url(forensic["fft"]["image"]) if forensic["fft"].get("image") is not None else "",
            "heatmap": heatmaps[0],
        }
        timeline = []
        analysis_type = "image"
        limitations = result["limitations"]
        top_suspicious_frames: list[dict] = []
    else:
        with tempfile.NamedTemporaryFile(suffix=suffix or ".mp4", delete=False) as temporary:
            temporary.write(raw)
            temp_path = temporary.name
        try:
            video = await asyncio.to_thread(analyze_video, temp_path)
            frames = video["frames"]
            if not frames:
                raise ValueError("No readable frames were found in the video.")
            primary = max(frames, key=lambda item: item["score"])["result"]
            forensic = await asyncio.to_thread(run_forensics, primary["image"], b"", filename, primary["face"].bbox)
            fusion = fuse(video["aggregate_score"], forensic, video["temporal_score"], video["limitations"])
            heatmaps = [heatmap_overlay(max(frames, key=lambda item: item["score"])["result"]["face"].crop)]
            visualizations = {
                "face_crop": _thumbnail(primary["face"].crop),
                "ela": image_to_data_url(forensic["ela"]["image"]) if forensic["ela"].get("image") is not None else "",
                "fft": image_to_data_url(forensic["fft"]["image"]) if forensic["fft"].get("image") is not None else "",
                "heatmap": heatmaps[0],
            }
            timeline = video["timeline"]
            analysis_type = "video"
            limitations = video["limitations"] + primary["limitations"]
            top_suspicious_frames = [
                {
                    "frame": int(item["frame"]),
                    "timestamp": float(item["timestamp"]),
                    "score": float(item["score"]),
                    "image": _thumbnail(item["image"]),
                    "heatmap": heatmap_overlay(item["result"]["face"].crop),
                }
                for item in sorted(frames, key=lambda item: item["score"], reverse=True)[:5]
            ]
        finally:
            Path(temp_path).unlink(missing_ok=True)
    selected = result if analysis_type == "image" else primary
    input_warnings = list(limitations)
    if forensic.get("_quality", {}).get("finding"):
        input_warnings.append(str(forensic["_quality"]["finding"]))
    evidence_payload = {
        **fusion,
        "per_model": selected["per_model"],
        "timeline": timeline,
        "limitations": limitations,
        "input_warnings": list(dict.fromkeys(input_warnings)),
    }
    summary_started = time.perf_counter()
    summary_result = await generate_summary(evidence_payload, client_key)
    timings = dict(result.get("timings", {}) if analysis_type == "image" else video.get("timings", {}))
    if analysis_type == "video":
        timings["primary_forensics_total"] = round(sum(float(value) for value in forensic.get("_timings", {}).values()), 2)
        for key, value in primary.get("timings", {}).items():
            timings[f"frame:{key}"] = round(float(value), 2)
    else:
        timings.update({f"forensics:{name}": value for name, value in forensic.get("_timings", {}).items()})
    timings["summary"] = round((time.perf_counter() - summary_started) * 1000, 2)
    timings["total"] = round((time.perf_counter() - started) * 1000, 2)
    LOGGER.info("Analysis timings (%s): %s", analysis_type, timings)
    no_face_detected = not selected["face"].found
    payload = {
        **evidence_payload,
        "id": report_id,
        "type": analysis_type,
        "heatmaps": heatmaps,
        "summary_markdown": summary_result.markdown,
        "processing_time": round(time.perf_counter() - started, 2),
        "metadata": {},
        "original_image": _thumbnail(selected["image"]),
        "reason_chips": _reason_chips(fusion["evidence"]),
        "no_face_detected": no_face_detected,
        "captions": CAPTIONS,
        "visualizations": visualizations,
        "summary_is_fallback": summary_result.is_fallback,
        "top_suspicious_frames": top_suspicious_frames,
        "timings": timings,
    }
    REPORTS[report_id] = payload
    save_report(report_id, payload)
    _cleanup_artifacts()
    return payload


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze(request: Request, file: UploadFile = File(...)) -> dict:
    _check_rate_limit(request)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            415,
            "Unsupported file type. Use JPG, PNG, WEBP, BMP, MP4, MOV, AVI, MKV, or WEBM.",
        )
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_UPLOAD_BYTES + 1024 * 1024:
                raise HTTPException(413, "Maximum upload size is 100 MB.")
        except ValueError as exc:
            raise HTTPException(400, "Invalid Content-Length header.") from exc
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Maximum upload size is 100 MB.")
        digest.update(chunk)
        chunks.append(chunk)
    cache_key = digest.hexdigest()
    if cache_key in CACHE:
        LOGGER.info("Cache hit for %s", cache_key)
        return CACHE[cache_key]
    if DISK_CACHE_ENABLED:
        cache_path = REPORT_DIR / f"{cache_key}.json"
        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                _cache_payload(cache_key, payload)
                LOGGER.info("Disk cache hit for %s", cache_key)
                return payload
            except (OSError, ValueError) as exc:
                LOGGER.warning("Ignoring invalid disk cache %s: %s", cache_path, exc)
    raw = b"".join(chunks)
    try:
        _validate_upload_content(raw, suffix)
        async with ANALYSIS_SEMAPHORE:
            payload = await _analyze(raw, file.filename or "upload.bin", _client_key(request))
        _cache_payload(cache_key, payload)
        if DISK_CACHE_ENABLED:
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            (REPORT_DIR / f"{cache_key}.json").write_text(json.dumps(payload, default=str), encoding="utf-8")
        return payload
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Analysis failed")
        raise HTTPException(500, "Analysis failed. Please try another file.") from exc


@app.get("/api/report/{report_id}")
async def report(report_id: str) -> FileResponse:
    from .config import REPORT_DIR

    path = REPORT_DIR / f"{report_id}.html"
    if not path.exists():
        raise HTTPException(404, "Report not found.")
    return FileResponse(path, media_type="text/html", filename=f"deepscan-{report_id}.html")
