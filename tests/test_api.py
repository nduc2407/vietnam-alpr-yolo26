from pathlib import Path
from fastapi.testclient import TestClient
from src.api import app

ROOT = Path(__file__).resolve().parents[1]


def test_api_health():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_api_invalid_file_rejected():
    with TestClient(app) as client:
        bad_resp = client.post("/predict", files={"file": ("fake.txt", b"plain text content", "text/plain")})
        assert bad_resp.status_code == 400
        assert "hop le" in bad_resp.json()["detail"]


def test_api_predict_valid_image():
    sample_img = next((ROOT / "data" / "biensoxe" / "test" / "images").glob("*.jpg"), None)
    if sample_img is None:
        sample_img = next((ROOT / "data" / "sample" / "images").glob("*.jpg"), None)

    if sample_img and sample_img.exists():
        with TestClient(app) as client:
            with open(sample_img, "rb") as f:
                resp = client.post("/predict", files={"file": (sample_img.name, f, "image/jpeg")})
            assert resp.status_code == 200
            data = resp.json()
            assert "plates" in data
            assert "timing_ms" in data
            assert "total" in data["timing_ms"]
