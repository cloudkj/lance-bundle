TODO
====

- [x] need to export the pooling strategy for the embedding model and write as part of a `manifest.json` into the bundle zip; at import time need to read this config then apply the right pooling in the SoverignBundle class
  - DONE: resolved by wrapping the SentenceTransformer modules and exporting entire graph inclusive of pooling/normalization
- [ ] CLI: extend `inspect` subcommand to view vector DB stats - # rows, distribution, indexes, etc
- [ ] CLI: extend to support loading HF dataset and performing same subcommands as local file
- [x] top level manifest with version and schema baked in for future compat
- [x] berkshire sharenletters as corpus 
- [x] export should include data provenance in manifest
- [x] Fix unit tests `pytest tests`
- [ ] dataset_name: str = None should be Optional[str] = None (or str | None on 3.10+) — the bare str = None annotation is technically incorrect per typing conventions even though Python won't enforce it.
- [ ] The assert os.path.exists(output_path) in __init__ and the length-check assert in _save_data are validation, not internal invariants — asserts get stripped under python -O, so these should probably be if ...: raise ValueError(...) instead. Worth fixing if this ever ships as a library others depend on.
- [ ] Forcing keyword-only args (def save(model, texts, vectors, output_path, *, dataset_name=None, ...)) would prevent accidental positional misordering even before/instead of the dataclass change — cheap, orthogonal improvement.
- [x] Inside the save_data function, you are constructing the payload by iterating over dataset.texts and dataset.vectors with a for loop to build a massive list of Python dictionaries.  In the data engineering space, doing row-by-row dictionary construction is a known performance killer.Because LanceDB operates natively on Apache Arrow, you should construct a PyArrow Table directly from the columnar lists (e.g., using pyarrow.Table.from_arrays). This will drastically reduce memory overhead and speed up the bundle creation process for large datasets.
- [ ] Remove manual parquet file conversion after HF dataset viewer <> Lance issues resolved. See: https://github.com/huggingface/dataset-viewer/pull/3395
- [ ] Extend export/builder.py to also create/return a `LanceBundle` object from `save`. This might require changing `LanceBundle` class to allow initialization from the actual internal objects - the ONNX ort_session object, the Lance table, and the tokenizer. This will allow the basic usage example to generate embeddings from some text, a SentenceTransformer, and in one flow output the bundle to disk and use it in-memory to do an embedding search