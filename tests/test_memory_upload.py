import tempfile

from fastapi.testclient import TestClient

from app.api import create_app


def test_large_upload_never_rolls_to_disk(monkeypatch, fake_resources):
    def forbidden(*args, **kwargs):
        raise AssertionError("User upload must not be written to disk")

    monkeypatch.setattr(tempfile.SpooledTemporaryFile, "rollover", forbidden)
    app = create_app(resource_loader=lambda settings: fake_resources,
                     retrieval_loader=lambda *args: None)
    with TestClient(app) as client:
        response = client.post("/predict-image", files={
            "file": ("invalid.png", b"x" * (2 * 1024 * 1024), "image/png")})
    assert response.status_code == 400
