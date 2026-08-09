import os
import json
import shutil
from typing import Optional, Union
import zipfile
import tempfile
import numpy as np
import lancedb
import onnxruntime as ort
from lance_bundle import constants
from lance_bundle.manifest import Manifest, ModelSignature
from tokenizers import Tokenizer

class LanceBundle:

    # Similarity/distance metric mapping between:
    # `sentence_transformers.util.similarity.SimilarityFunction`
    # `lancedb.query.LanceVectorQueryBuilder.distance_type`
    # See: https://sbert.net/docs/package_reference/util/similarity.html#sentence_transformers.util.similarity.SimilarityFunction
    # See: https://lancedb.github.io/lancedb/python/python/#lancedb.query.LanceVectorQueryBuilder.distance_type
    _SIMILARITY_FUNCTION_TO_DISTANCE_TYPE = {
        "cosine": "cosine",
        "dot": "dot",
        "euclidean": "l2",
        "manhattan": None,
    }
    _DEFAULT_DISTANCE_TYPE = "cosine"

    def __init__(self, path: Union[str, tempfile.TemporaryDirectory]):
        """
        Initializes a LanceBundle.
        Accepts a string path (directory or zip) OR a TemporaryDirectory object.
        """
        # Maintain reference to ensure temporary directory is cleaned up upon garbage collection of `self`
        base_path, self._temp_dir = self._resolve_base_path(path)

        data_dir = os.path.join(base_path, constants.DATA_DIR)
        model_dir = os.path.join(base_path, constants.MODEL_DIR)

        self._manifest = self._load_manifest(base_path)
        signature = self._manifest.model.signature

        # Load vector search distance type; default to cosine as fallback
        self._distance_type = self._SIMILARITY_FUNCTION_TO_DISTANCE_TYPE.get(
            signature.distance_metric,
            self._DEFAULT_DISTANCE_TYPE
        )

        self._tokenizer = self._load_tokenizer(model_dir, signature)
        self._ort_session = self._load_model(model_dir)
        self._table = self._load_data(data_dir)

    @staticmethod
    def _resolve_base_path(
        path: Union[str, tempfile.TemporaryDirectory]
    ) -> tuple[str, Optional[tempfile.TemporaryDirectory]]:
        """
        Resolves a bundle path (directory, zip file, or TemporaryDirectory) to a base
        directory. Returns (base_path, temp_dir), where temp_dir is the
        TemporaryDirectory that must be kept alive for the lifetime of the bundle
        (None if the given path is a plain directory that needs no cleanup).
        """
        if isinstance(path, tempfile.TemporaryDirectory):
            return path.name, path
        if os.path.isdir(path):
            return path, None
        if zipfile.is_zipfile(path):
            temp_dir = tempfile.TemporaryDirectory()
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir.name)
            return temp_dir.name, temp_dir
        raise ValueError(f"Error: {path} must be a directory, zip file, or TemporaryDirectory")

    @staticmethod
    def _load_manifest(base_path: str) -> Manifest:
        with open(os.path.join(base_path, constants.MANIFEST_PATH), "r") as file:
            return Manifest.from_json(json.load(file))

    @staticmethod
    def _load_tokenizer(model_dir: str, signature: ModelSignature) -> Tokenizer:
        tokenizer = Tokenizer.from_file(os.path.join(model_dir, constants.TOKENIZER_PATH))
        tokenizer.enable_padding()
        if signature.max_seq_length is not None:
            tokenizer.enable_truncation(max_length=signature.max_seq_length)
        return tokenizer

    @staticmethod
    def _load_model(model_dir: str) -> ort.InferenceSession:
        return ort.InferenceSession(os.path.join(model_dir, constants.MODEL_ONNX_PATH))

    @staticmethod
    def _load_data(data_dir: str) -> lancedb.table.Table:
        return lancedb.connect(data_dir).open_table(constants.DATA_TABLE_NAME)

    def _embed(self, text: str) -> list[float]:
        # TODO: inspect if this method needs to be symmetric to the ONNX export in BundleBuilder (it appears so)
        encoded = self._tokenizer.encode(text)
        
        inputs = {
            "input_ids": np.array([encoded.ids], dtype=np.int64),
            "attention_mask": np.array([encoded.attention_mask], dtype=np.int64)
        }

        if "token_type_ids" in [i.name for i in self._ort_session.get_inputs()]:
             inputs["token_type_ids"] = np.array([encoded.type_ids], dtype=np.int64)

        # The ONNX session does the transformer math, the pooling, AND the normalization!
        outputs = self._ort_session.run(None, inputs)
        
        # We just grab the final vector and return it. No Python math required.
        return outputs[0][0].tolist()

    def metadata(self) -> dict:
        table, schema = self._table, self._table.schema
        return {
            "manifest": self._manifest.to_json(),
            "data": {
                "metadata": {k.decode(): v.decode() for k, v in (schema.metadata or {}).items()},
                "schema": {f.name: str(f.type) for f in schema},
                "rows": table.count_rows(),
            }
        }

    def search(self, query: str, limit: int=5):
        vector = self._embed(query)
        query_builder = self._table.search(vector).distance_type(self._distance_type).limit(limit)
        return query_builder.to_list()

def _download_snapshot(dataset_name: str, dest_dir: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError("load_dataset requires the 'hub' extra: pip install lance-bundle[hub]") from e
    snapshot_download(
        repo_id=dataset_name,
        repo_type="dataset",
        local_dir=dest_dir,
    )

def load(bundle_path: str) -> LanceBundle:
    """
    Loads a LanceBundle from a local path.

    Args:
        bundle_path: Path to a bundle directory or a bundle `.zip` archive.

    Returns:
        The loaded LanceBundle, ready to search.
    """
    return LanceBundle(bundle_path)

def load_dataset(dataset_name: str, output_path: Optional[str] = None) -> LanceBundle:
    """
    Downloads a bundle dataset from the Hugging Face Hub and loads it as a LanceBundle.

    Requires the 'hub' extra (`pip install lance-bundle[hub]`).

    See: https://huggingface.co/lance-bundle/datasets

    Args:
        dataset_name: Hugging Face Hub dataset repo id, e.g. "org/dataset-name".
        output_path: If given, the downloaded bundle contents are also archived as a `.zip` file at this path.

    Returns:
        The downloaded and loaded LanceBundle, ready to search.
    """
    temp_dir = tempfile.TemporaryDirectory()
    try:
        _download_snapshot(dataset_name, temp_dir.name)
        # Load the bundle before archiving, so a broken download never gets persisted
        bundle = LanceBundle(temp_dir)
        if output_path is not None:
            base_name, _ = os.path.splitext(output_path)
            shutil.make_archive(base_name, 'zip', temp_dir.name)
    except Exception:
        temp_dir.cleanup()
        raise
    return bundle
