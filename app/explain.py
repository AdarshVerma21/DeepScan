from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image

from .config import RESPONSE_IMAGE_MAX_PX, RESPONSE_IMAGE_QUALITY


def image_to_data_url(image: np.ndarray) -> str:
    height, width = image.shape[:2]
    scale = min(1.0, RESPONSE_IMAGE_MAX_PX / max(height, width))
    if scale < 1:
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="WEBP", quality=RESPONSE_IMAGE_QUALITY, method=4)
    return "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def heatmap_overlay(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    saliency = cv2.normalize(cv2.Laplacian(gray, cv2.CV_32F), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
    return image_to_data_url(cv2.addWeighted(image, 0.6, heat, 0.4, 0))
