import os
import zipfile
from lance_bundle import save, ExportDataset
from lance_bundle import constants

def test_save_bundle_creates_valid_archive(dummy_model, dummy_data, temp_bundle_path):
    texts, vectors = dummy_data

    # Execute the export pipeline
    save(dummy_model, ExportDataset(texts, vectors), temp_bundle_path)
    
    # 1. Verify the file was created
    assert os.path.exists(temp_bundle_path)
    assert temp_bundle_path.endswith(".zip")
    
    # 2. Verify the internal structure of the Zip archive
    with zipfile.ZipFile(temp_bundle_path, 'r') as zip_ref:
        files = zip_ref.namelist()
        
        # Check that the LanceDB directory exists
        assert any(f.startswith(f"{constants.DATA_DIR}") for f in files), "LanceDB data directory missing"
        
        # Check that ONNX and Tokenizer artifacts exist
        assert f"{constants.MODEL_DIR}{constants.MODEL_ONNX_PATH}" in files, "ONNX model missing"
        assert f"{constants.MODEL_DIR}tokenizer.json" in files, "Tokenizer missing"
        assert f"{constants.MANIFEST_PATH}" in files, "Manifest missing"
