import os
import sys
import zipfile
import pytest
import lance_bundle.runtime as runtime
from lance_bundle import load, load_dataset, save, ExportDataset
from lance_bundle.export import LanceBundleBuilder

@pytest.fixture
def fake_download_snapshot(monkeypatch, dummy_model, dummy_data):
    """Stubs out the HF Hub network call: builds a real bundle directly into
    dest_dir (mirroring what a real snapshot_download would leave on disk),
    so load_dataset() can be tested without any network access."""
    texts, vectors = dummy_data

    def _fake_download(dataset_name, dest_dir):
        LanceBundleBuilder(dest_dir).build(dummy_model, ExportDataset(texts, vectors))

    monkeypatch.setattr(runtime, "_download_snapshot", _fake_download)

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

def test_load_dataset_downloads_and_loads(fake_download_snapshot, dummy_data):
    texts, _ = dummy_data

    bundle = load_dataset("fake-org/fake-dataset")

    assert bundle.metadata()["data"]["rows"] == len(texts)
    results = bundle.search("Dummy query for plumbing test", limit=len(texts))
    assert len(results) == len(texts)

def test_load_dataset_writes_output_path(fake_download_snapshot, dummy_data, tmp_path):
    texts, _ = dummy_data
    output_path = str(tmp_path / "downloaded.zip")

    bundle = load_dataset("fake-org/fake-dataset", output_path=output_path)
    assert bundle.metadata()["data"]["rows"] == len(texts)

    # The archived zip is written and independently loadable
    assert os.path.exists(output_path)
    archived_bundle = load(output_path)
    assert archived_bundle.metadata()["data"]["rows"] == len(texts)

def test_download_snapshot_requires_hub_extra(monkeypatch):
    # Force HuggingFace dependecy import to fail
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    with pytest.raises(ImportError, match="hub"):
        runtime._download_snapshot("fake-org/fake-dataset", "/tmp/unused")
