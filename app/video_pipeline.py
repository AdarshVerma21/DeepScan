from __future__ import annotations

import cv2
import numpy as np
import time

from .config import (
    MAX_VIDEO_SECONDS,
    VIDEO_BATCH_SIZE,
    VIDEO_EARLY_STOP_HIGH,
    VIDEO_EARLY_STOP_LOW,
    VIDEO_EARLY_STOP_MIN_FRAMES,
    VIDEO_FRAME_MAX_PX,
    VIDEO_MAX_FRAMES,
)
from .image_pipeline import analyze_images_batch


def _frame_score(result: dict[str, object]) -> float:
    ensemble_score = result.get("ensemble_score")
    if ensemble_score is None:
        return 0.5
    return float(np.clip(float(ensemble_score), 0.0, 1.0))


def analyze_video(path: str) -> dict[str, object]:
    started = time.perf_counter()
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        capture.release()
        raise ValueError("The uploaded video could not be opened by the available codec.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_duration = count / fps if count > 0 else 0
    analyzed_count = min(count, max(1, int(MAX_VIDEO_SECONDS * fps)))
    duration = analyzed_count / fps if analyzed_count else 0
    target = min(VIDEO_MAX_FRAMES, max(2, int(duration * 2)))
    indices = sorted(set(np.linspace(0, max(0, analyzed_count - 1), target).astype(int).tolist() + [0, max(0, analyzed_count - 1)]))
    frames = []
    sampled_images = []
    sampled_meta = []
    selected = set(indices)
    batch_size = max(1, VIDEO_BATCH_SIZE)
    model_batch_ms = 0.0

    def process_batch() -> bool:
        nonlocal model_batch_ms
        if not sampled_images:
            return False
        batch_images = list(sampled_images)
        batch_meta = list(sampled_meta)
        sampled_images.clear()
        sampled_meta.clear()
        model_started = time.perf_counter()
        results = analyze_images_batch(batch_images)
        model_batch_ms += (time.perf_counter() - model_started) * 1000
        for (frame_index, frame), result in zip(batch_meta, results):
            frames.append({"frame": frame_index, "timestamp": frame_index / fps, "score": _frame_score(result), "image": frame, "result": result})
        scores = [float(item["score"]) for item in frames]
        if len(scores) >= VIDEO_EARLY_STOP_MIN_FRAMES:
            high = all(score >= VIDEO_EARLY_STOP_HIGH for score in scores[-VIDEO_EARLY_STOP_MIN_FRAMES:])
            low = all(score <= VIDEO_EARLY_STOP_LOW for score in scores[-VIDEO_EARLY_STOP_MIN_FRAMES:])
            return high or low
        return False

    index = 0
    early_stopped = False
    while index <= max(0, analyzed_count - 1):
        if not capture.grab():
            break
        if index in selected:
            ok, frame = capture.retrieve()
            if ok:
                height, width = frame.shape[:2]
                if max(height, width) > VIDEO_FRAME_MAX_PX:
                    scale = VIDEO_FRAME_MAX_PX / max(height, width)
                    frame = cv2.resize(
                        frame,
                        (max(1, int(width * scale)), max(1, int(height * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                sampled_images.append(frame)
                sampled_meta.append((index, frame))
                if len(sampled_images) >= batch_size:
                    early_stopped = process_batch()
                    if early_stopped:
                        break
        index += 1
    capture.release()
    if not early_stopped:
        process_batch()
    decode_ms = max(0.0, (time.perf_counter() - started) * 1000 - model_batch_ms)
    scores = [float(item["score"]) for item in frames]
    mean_score = float(np.mean(scores)) if scores else 0.5
    top_count = max(2, int(np.ceil(len(scores) * 0.2))) if scores else 1
    top_percentile = float(np.mean(sorted(scores, reverse=True)[:top_count])) if scores else 0.5
    consistency = (
        float(np.clip(1.0 - np.std(scores) * 2.0, 0.0, 1.0))
        if len(scores) > 1
        else 0.5
    )
    aggregate = 0.65 * mean_score + 0.35 * top_percentile
    temporal = float(np.clip(top_percentile * (0.7 + 0.3 * consistency), 0.0, 1.0))
    limitations = ["Audio/lip-sync proxy is skipped unless a separate audio extractor is available."]
    if source_duration > MAX_VIDEO_SECONDS:
        limitations.append(
            f"Only the first {MAX_VIDEO_SECONDS} seconds of a {source_duration:.1f}-second video were analyzed."
        )
    if VIDEO_MAX_FRAMES < 60 or VIDEO_FRAME_MAX_PX < 1536:
        limitations.append(
            f"Video frames were bounded to at most {VIDEO_MAX_FRAMES} samples and "
            f"{VIDEO_FRAME_MAX_PX}px for this deployment."
        )
    if early_stopped:
        limitations.append("Frame analysis stopped early after sustained agreement between sampled frames.")
    return {
        "frames": frames,
        "timeline": [{"frame": int(item["frame"]), "timestamp": float(item["timestamp"]), "score": round(item["score"], 4)} for item in frames],
        "aggregate_score": aggregate,
        "top_percentile_score": top_percentile,
        "temporal_consistency": consistency,
        "temporal_score": temporal,
        "limitations": limitations,
        "timings": {
            "video_decode": round(decode_ms, 2),
            "video_model_batches": round(model_batch_ms, 2),
            "video_frames_analyzed": float(len(frames)),
            "video_early_stopped": float(early_stopped),
        },
    }
