import json
import os
import zipfile
import lancedb
import pytest
from lance_bundle import constants, save, ExportDataset

def test_save_bundle_creates_valid_archive(dummy_model, dummy_data, tmp_path):
    texts, vectors = dummy_data
    bundle_path = str(tmp_path / "bundle.zip")

    # Execute the export pipeline
    save(dummy_model, ExportDataset(texts, vectors), bundle_path)

    # 1. Verify the file was created
    assert os.path.exists(bundle_path)
    assert bundle_path.endswith(".zip")

    # 2. Verify the internal structure of the Zip archive
    with zipfile.ZipFile(bundle_path, 'r') as zip_ref:
        files = zip_ref.namelist()
        
        # Check that the LanceDB directory exists
        assert any(f.startswith(f"{constants.DATA_DIR}") for f in files), "LanceDB data directory missing"
        
        # Check that ONNX and Tokenizer artifacts exist
        assert f"{constants.MODEL_DIR}{constants.MODEL_ONNX_PATH}" in files, "ONNX model missing"
        assert f"{constants.MODEL_DIR}tokenizer.json" in files, "Tokenizer missing"
        assert f"{constants.MANIFEST_PATH}" in files, "Manifest missing"

def test_save_bundle_manifest_content(dummy_model, dummy_data, tmp_path):
    texts, vectors = dummy_data
    bundle_path = str(tmp_path / "bundle.zip")

    save(dummy_model, ExportDataset(texts, vectors), bundle_path)

    with zipfile.ZipFile(bundle_path, 'r') as zip_ref:
        manifest = json.loads(zip_ref.read(constants.MANIFEST_PATH))

    # Top-level manifest fields
    assert manifest["version"], "Manifest schema version missing"
    assert manifest["generator"].startswith("lance_bundle=="), "Unexpected generator format"
    assert manifest["created_at"], "Manifest created_at missing"

    # Model component
    model_component = manifest["components"]["model"]
    assert model_component["type"] == "onnx"
    assert model_component["path"] == constants.MODEL_DIR
    assert model_component["generator"].startswith("sentence_transformers==")
    assert model_component["signature"]["max_seq_length"] == 128

    # Data component
    data_component = manifest["components"]["data"]
    assert data_component["type"] == "lancedb"
    assert data_component["path"] == constants.DATA_DIR

def test_save_bundle_data_roundtrip(dummy_model, dummy_data, tmp_path):
    texts, vectors = dummy_data
    metadata = [{"topic": f"dummy-topic-{i}"} for i in range(len(texts))]
    dataset = ExportDataset(
        texts,
        vectors,
        metadata=metadata,
        name="test-dataset",
        description="A dataset used for testing.",
        source="https://example.com/source",
    )
    bundle_path = str(tmp_path / "bundle.zip")

    save(dummy_model, dataset, bundle_path)

    with zipfile.ZipFile(bundle_path, 'r') as zip_ref:
        manifest = json.loads(zip_ref.read(constants.MANIFEST_PATH))
        zip_ref.extractall(tmp_path / "extracted")

    # Manifest reflects dataset provenance
    data_component = manifest["components"]["data"]
    assert data_component["dataset_name"] == dataset.name
    assert data_component["dataset_description"] == dataset.description
    assert data_component["dataset_source"] == dataset.source

    # Per-row metadata is preserved in the LanceDB table
    db = lancedb.connect(str(tmp_path / "extracted" / constants.DATA_DIR))
    table = db.open_table(constants.DATA_TABLE_NAME)
    arrow_table = table.to_arrow()
    assert arrow_table.num_rows == len(texts)
    assert arrow_table["text"].to_pylist() == texts
    for actual_vector, expected_vector in zip(arrow_table["vector"].to_pylist(), vectors):
        assert actual_vector == pytest.approx(expected_vector)

def test_export_dataset_preconditions(dummy_data):
    with pytest.raises(ValueError):
        ExportDataset([], [])

    texts, vectors = dummy_data
    with pytest.raises(ValueError):
        ExportDataset(texts, vectors[:-1])

    with pytest.raises(ValueError):
        ExportDataset(texts, vectors, metadata=[{}])
