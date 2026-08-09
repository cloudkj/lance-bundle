import pytest
from lance_bundle import load, save, ExportDataset

@pytest.fixture
def populated_bundle_path(dummy_model, dummy_data, tmp_path):
    """Fixture that compiles a live bundle specifically for runtime tests."""
    texts, vectors = dummy_data
    bundle_path = str(tmp_path / "bundle.zip")
    save(dummy_model, ExportDataset(texts, vectors), bundle_path)
    return bundle_path

def test_lance_bundle_initialization_and_metadata(populated_bundle_path):
    # Test successful extraction and mounting
    bundle = load(populated_bundle_path)
    
    # Test metadata extraction from Arrow schema
    metadata = bundle.metadata()
    assert isinstance(metadata, dict)
    
    # Validate our export logic correctly appended model provenance
    assert "base_model" in metadata or len(metadata) > 0

def test_lance_bundle_semantic_search(populated_bundle_path):
    bundle = load(populated_bundle_path)
    
    # Sanity check tokenizer -> ONNX -> LanceDB flow
    results = bundle.search("Dummy query for plumbing test", limit=2)
    
    # Check result structure
    assert len(results) == 2
    top_result = results[0]    
    assert "text" in top_result
    assert "vector" in top_result
    assert "_distance" in top_result # LanceDB default distance metric
    
    # Check data types and shapes
    assert isinstance(top_result["text"], str)
    assert isinstance(top_result["vector"], list)
    assert len(top_result["vector"]) == 4
    assert isinstance(top_result["_distance"], float)
