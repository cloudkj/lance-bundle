import zipfile
from lance_bundle import load, save, ExportDataset

def test_load_bundle_from_directory(dummy_model, dummy_data, tmp_path):
    texts, vectors = dummy_data
    bundle_path = str(tmp_path / "bundle.zip")
    save(dummy_model, ExportDataset(texts, vectors), bundle_path)

    extracted_dir = tmp_path / "extracted"
    with zipfile.ZipFile(bundle_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_dir)

    # Load directly from the extracted directory, not the zip
    bundle = load(str(extracted_dir))

    assert bundle.metadata()["data"]["rows"] == len(texts)
    results = bundle.search("Dummy query for plumbing test", limit=len(texts))
    assert len(results) == len(texts)

def test_load_bundle_metadata_content(dummy_model, dummy_data, tmp_path):
    texts, vectors = dummy_data
    dataset = ExportDataset(
        texts,
        vectors,
        name="test-dataset",
        description="A dataset used for testing.",
        source="https://example.com/source",
    )
    bundle_path = str(tmp_path / "bundle.zip")
    save(dummy_model, dataset, bundle_path)

    bundle = load(bundle_path)
    metadata = bundle.metadata()

    # Manifest content reflects dataset provenance
    data_component = metadata["manifest"]["components"]["data"]
    assert data_component["dataset_name"] == dataset.name
    assert data_component["dataset_description"] == dataset.description
    assert data_component["dataset_source"] == dataset.source

    # LanceDB table schema and row count are surfaced correctly
    assert "text" in metadata["data"]["schema"]
    assert "vector" in metadata["data"]["schema"]
    assert metadata["data"]["rows"] == len(texts)

def test_load_bundle_search_returns_metadata_when_present(dummy_model, dummy_data, tmp_path):
    texts, vectors = dummy_data
    metadata = [{"topic": f"dummy-topic-{i}"} for i in range(len(texts))]
    dataset = ExportDataset(texts, vectors, metadata=metadata)
    bundle_path = str(tmp_path / "bundle.zip")
    save(dummy_model, dataset, bundle_path)

    bundle = load(bundle_path)
    results = bundle.search("Dummy query for plumbing test", limit=len(texts))

    expected_metadata_by_text = dict(zip(texts, metadata))
    assert len(results) == len(texts)
    for result in results:
        assert "text" in result
        assert "vector" in result
        assert "_distance" in result
        assert "metadata" in result
        assert isinstance(result["text"], str)
        assert isinstance(result["vector"], list)
        assert isinstance(result["_distance"], float)
        assert len(result["vector"]) == 4
        assert result["metadata"] == expected_metadata_by_text[result["text"]]
