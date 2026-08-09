import importlib.metadata
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
import sentence_transformers
from sentence_transformers import SentenceTransformer
from .. import constants
from ..manifest import DataComponent, Manifest, ModelComponent, ModelSignature
from .data import ExportDataset, save_data
from .model import save_model

_PACKAGE_NAME = "lance_bundle"
_MANIFEST_SCHEMA_VERSION = "0.0.1"

class LanceBundleBuilder:

    def __init__(self, output_path: str):
        assert os.path.exists(output_path), "Bundle output path must exist"
        self._root_path = output_path
        self._data_path = os.path.join(output_path, constants.DATA_DIR)
        self._model_path = os.path.join(output_path, constants.MODEL_DIR)
        self._model_onnx_path = os.path.join(self._model_path, constants.MODEL_ONNX_PATH)

    def _build_manifest(self, model: SentenceTransformer, dataset: ExportDataset) -> Manifest:
        model_component = ModelComponent(
            type="onnx",
            path=constants.MODEL_DIR,
            generator=f"{sentence_transformers.__name__}=={importlib.metadata.version(sentence_transformers.__name__)}",
            model_card=model.model_card_data.to_dict(),
            signature=ModelSignature(
                distance_metric=model.similarity_fn_name or None,
                max_seq_length=model.get_max_seq_length() or None,
            ),
        )

        data_component = DataComponent(
            type="lancedb",
            path=constants.DATA_DIR,
            dataset_name=dataset.name,
            dataset_description=dataset.description,
            dataset_source=dataset.source,
        )

        return Manifest(
            version=_MANIFEST_SCHEMA_VERSION,
            generator=f"{_PACKAGE_NAME}=={importlib.metadata.version(_PACKAGE_NAME)}",
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            model=model_component,
            data=data_component,
        )

    def _write_manifest(self, manifest: Manifest):
        with open(os.path.join(self._root_path, constants.MANIFEST_PATH), "w") as file:
            json.dump(manifest.to_json(), file, default=str, indent=4)

    def build(self, model: SentenceTransformer, dataset: ExportDataset):
        save_data(self._data_path, dataset)
        save_model(model, self._model_path, self._model_onnx_path)
        manifest = self._build_manifest(model, dataset)
        self._write_manifest(manifest)

def save(model: SentenceTransformer, dataset: ExportDataset, output_path: str):
    with tempfile.TemporaryDirectory() as temp_dir_path:
        LanceBundleBuilder(temp_dir_path).build(model, dataset)
        base_name, _ = os.path.splitext(output_path)
        shutil.make_archive(base_name, 'zip', temp_dir_path)
