from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)
_FACE_DETECTOR = None
_FACE_MESH = None


@dataclass
class FaceResult:
    crop: np.ndarray
    bbox: tuple[int, int, int, int]
    landmarks: list[tuple[float, float, float]]
    found: bool


def _crop_with_margin(image: np.ndarray, bbox: tuple[int, int, int, int], margin: float) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    left = max(0, int(x1 - margin * box_width))
    top = max(0, int(y1 - margin * box_height))
    right = min(width, int(x2 + margin * box_width))
    bottom = min(height, int(y2 + margin * box_height))
    crop = image[top:bottom, left:right]
    return crop if crop.size else image


def face_variant(image: np.ndarray, face: FaceResult, margin: float) -> np.ndarray:
    return _crop_with_margin(image, face.bbox, margin)


def detect_face(image: np.ndarray, margin: float = 0.20) -> FaceResult:
    """Detect one face; use the complete image when detection is unavailable."""
    height, width = image.shape[:2]
    try:
        global _FACE_DETECTOR
        if _FACE_DETECTOR is None:
            import mediapipe as mp
            _FACE_DETECTOR = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.45
            )
        result = _FACE_DETECTOR.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if result.detections:
            box = result.detections[0].location_data.relative_bounding_box
            x1 = max(0, int((box.xmin - margin * box.width) * width))
            y1 = max(0, int((box.ymin - margin * box.height) * height))
            x2 = min(width, int((box.xmin + box.width * (1 + margin)) * width))
            y2 = min(height, int((box.ymin + box.height * (1 + margin)) * height))
            crop = image[y1:y2, x1:x2]
            if crop.size:
                return FaceResult(crop, (x1, y1, x2, y2), [], True)
    except Exception as exc:
        LOGGER.warning("MediaPipe face detection unavailable: %s", exc)
    return FaceResult(image, (0, 0, width, height), [], False)


def landmarks_for_frame(image: np.ndarray) -> list[tuple[float, float, float]]:
    """Return normalized Face Mesh landmarks when MediaPipe is installed."""
    try:
        global _FACE_MESH
        if _FACE_MESH is None:
            import mediapipe as mp
            _FACE_MESH = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1, refine_landmarks=True
            )
        result = _FACE_MESH.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if result.multi_face_landmarks:
            return [
                (point.x, point.y, point.z)
                for point in result.multi_face_landmarks[0].landmark
            ]
    except Exception as exc:
        LOGGER.debug("Face mesh unavailable: %s", exc)
    return []
