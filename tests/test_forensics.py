import io

import cv2
import numpy as np
from PIL import Image

from app.forensics import (
    error_level_analysis,
    fft_analysis,
    face_boundary,
    metadata_check,
    noise_residual,
)


def test_forensic_scores_are_normalized():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:48, 16:48] = 255
    for result in (error_level_analysis(image), fft_analysis(image), face_boundary(image, (8, 8, 56, 56))):
        assert 0 <= result["score"] <= 1


def test_ela_returns_visualization():
    image = np.full((32, 32, 3), 120, dtype=np.uint8)
    assert error_level_analysis(image)["image"] is not None


def test_noise_residual_does_not_saturate_on_flat_background():
    image = np.full((128, 128, 3), 128, dtype=np.uint8)
    image[32:96, 32:96] = 140

    result = noise_residual(image, (32, 32, 96, 96))

    assert 0 <= result["score"] <= 1
    assert result["available"] is False
    assert "unreliable" in result["finding"]


def test_noise_residual_is_bounded_for_textured_input():
    image = np.random.default_rng(7).integers(
        0, 256, size=(128, 128, 3), dtype=np.uint8
    )

    result = noise_residual(image, (32, 32, 96, 96))

    assert 0 <= result["score"] <= 1
    assert result["available"] is True


def test_missing_exif_is_not_evidence_of_manipulation():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (120, 130, 140)).save(buffer, format="PNG")

    result = metadata_check(buffer.getvalue(), "sample.png")

    assert result["score"] == 0.5
    assert result["available"] is False
    assert "absence is not evidence" in result["finding"]


def test_invalid_noise_bbox_is_unavailable():
    result = noise_residual(np.zeros((32, 32, 3), dtype=np.uint8), (40, 40, 20, 20))

    assert result["available"] is False
