import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.public import public_profile
from app.resources import OFFICIAL_LABELS, ResourceValidationError, parse_label_mapping
from conftest import image_bytes


def test_profile_allowlist_and_copy():
    character = {
        "deskripsi": "Profil publik", "ciri_ciri": ["mahkota"],
        "gambar_utama": "dataset_asli/jayangrana/1.PNG",
        "gambar_training": ["/content/private"], "gambar_asli": ["private"],
        "jumlah_gambar_training": 24, "embedding": [1], "metadata_training": {},
    }
    result = public_profile(character)
    assert set(result) == {
        "deskripsi", "deskripsi_lengkap", "filosofi", "ciri_ciri", "kata_kunci", "gambar_utama"
    }
    result["ciri_ciri"].append("baru")
    assert character["ciri_ciri"] == ["mahkota"]


@pytest.mark.parametrize("image", ["/content/drive/private.png", "C:/private.png",
    "dataset_asli/../private.png", "dataset_tryning/0/a.png", "https://x/a?token=secret"])
def test_internal_image_not_public(image):
    assert "gambar_utama" not in public_profile({"gambar_utama": image})


def test_prediction_uses_public_serializer(settings, fake_resources):
    fake_resources.characters[19]["embedding"] = [99]
    with TestClient(create_app(settings, lambda _: fake_resources)) as client:
        payload = client.post("/predict-image", files={
            "file": ("sample.png", image_bytes("PNG"), "image/png")
        }).json()
    assert "embedding" not in payload["profil"]
    assert payload["prediksi"]["id"] == 19


@pytest.mark.parametrize("reverse", [False, True])
def test_nested_legacy_mapping(reverse):
    by_id = {str(i): label for i, label in enumerate(OFFICIAL_LABELS)}
    by_label = {label: i for i, label in enumerate(OFFICIAL_LABELS)}
    raw = {"id2label": by_label if reverse else by_id,
           "label2id": by_id if reverse else by_label}
    assert parse_label_mapping(raw) == dict(enumerate(OFFICIAL_LABELS))


def test_nested_mapping_rejects_conflicting_official_order():
    by_id = {str(i): label for i, label in enumerate(OFFICIAL_LABELS)}
    bad = dict(by_id, **{"0": "gunungan"})
    with pytest.raises(ResourceValidationError):
        parse_label_mapping({"id2label": by_id, "label2id": bad})
