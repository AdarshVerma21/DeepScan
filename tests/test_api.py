from fastapi.testclient import TestClient

from app.main import app


def test_health():
    response = TestClient(app).get("/api/health")
    assert response.status_code == 200
    assert response.json()["service"] == "DeepScan"


def test_security_headers():
    response = TestClient(app).get("/api/health")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


def test_unsupported_upload_extension_is_rejected():
    response = TestClient(app).post(
        "/api/analyze",
        files={"file": ("payload.exe", b"not media", "application/octet-stream")},
    )
    assert response.status_code == 415


def test_mismatched_image_content_is_rejected():
    response = TestClient(app).post(
        "/api/analyze",
        files={"file": ("image.jpg", b"not an image", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "valid image" in response.json()["detail"]


def test_ready_reports_available_detectors():
    response = TestClient(app).get("/api/ready")
    assert response.status_code in {200, 503}
    if response.status_code == 200:
        assert response.json()["models"]["available"] > 0
