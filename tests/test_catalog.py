from app.catalog import Catalog


def test_catalog_uses_json_names_and_id_order(fake_resources):
    fake_resources.characters[10]["nama"] = "Darun Dia"
    fake_resources.characters[0]["gambar_utama"] = "dataset_asli/jayangrana/1.PNG"
    fake_resources.characters[0]["embedding"] = [999]
    catalog = Catalog(fake_resources.labels, dict(reversed(list(fake_resources.characters.items()))))
    assert [item.id for item in catalog.items] == list(range(20))
    assert catalog.items[10].nama == "Darun Dia"
    assert catalog.items[0].gambar_utama == "dataset_asli/jayangrana/1.PNG"
    assert "embedding" not in catalog.items[0].model_dump()


def test_catalog_bypasses_semantic_and_contains_followup():
    from app.retrieval import RetrievalService
    from evaluation.mock_search import mock_service
    service = RetrievalService(mock_service().catalog)
    result = service.search("Coba berikan saya contoh tokoh Wayang Saksak?")
    assert result.mode == "catalog"
    assert len(result.catalog) == 20
    assert result.suggestions and "pilih" in result.message.lower()
