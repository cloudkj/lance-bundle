from dataclasses import dataclass
from typing import Optional
import lancedb
import numpy as np
import pyarrow as pa
from .. import constants

@dataclass
class ExportDataset:
    texts: list[str]
    vectors: list[list[float]]
    metadata: Optional[list[dict]] = None
    name: Optional[str] = None
    description: Optional[str] = None
    source: Optional[str] = None

    def __post_init__(self):
        if not (len(self.texts) > 0 and len(self.texts) == len(self.vectors)):
            raise ValueError("texts and vectors must be non-empty and of equal length")
        if self.metadata is not None and len(self.metadata) != len(self.texts):
            raise ValueError("metadata, if provided, must be the same length as texts and vectors")

def save_data(data_path: str, dataset: ExportDataset):
    # Build the payload columnar-first
    vectors_np = np.asarray(dataset.vectors, dtype=np.float32)
    vector_array = pa.FixedSizeListArray.from_arrays(
        pa.array(vectors_np.reshape(-1), type=pa.float32()), vectors_np.shape[1]
    )
    arrays = [pa.array(dataset.texts), vector_array]
    names = ["text", "vector"]
    if dataset.metadata is not None:
        arrays.append(pa.array(dataset.metadata))
        names.append("metadata")
    arrow_data = pa.Table.from_arrays(arrays, names=names)

    # Output to LanceDB
    db = lancedb.connect(data_path)
    table = db.create_table(constants.DATA_TABLE_NAME, data=arrow_data)

    # Overwrite the table to cement the new metadata schema
    db.create_table(constants.DATA_TABLE_NAME, data=table.to_arrow(), mode="overwrite")
