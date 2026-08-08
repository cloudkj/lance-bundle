from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import tempfile
import urllib.request
from lance_bundle import save, ExportDataset

url = "https://raw.githubusercontent.com/run-llama/llama_index/refs/heads/main/docs/examples/data/paul_graham/paul_graham_essay.txt"
with tempfile.NamedTemporaryFile() as tmp:
    with urllib.request.urlopen(url) as response:
        while chunk := response.read(8192):
            tmp.write(chunk)
    documents = SimpleDirectoryReader(input_files=[tmp.name]).load_data()

print(f"Loaded documents {len(documents)=}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = HuggingFaceEmbedding(
    model_name=model_name
)

print(f"HF model card {embed_model._model.model_card_data}")

index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, show_progress=True)
docstore = index.docstore
vector_store = index.vector_store

node_ids = [node_id for _, ref_doc_info in docstore.get_all_ref_doc_info().items() for node_id in ref_doc_info.node_ids]
texts = [docstore.get_node(node_id).text for node_id in node_ids]
embeddings = [vector_store.get(node_id) for node_id in node_ids]

print(f"{len(texts)=}, {len(embeddings)=}")

# Export
save(
    embed_model._model,
    ExportDataset(
        texts,
        embeddings,
        metadata=[{"node_id": node_id} for node_id in node_ids],
        description="lance-bundle LlamaIndex export example",
    ),
    "examples/pg_essay_embeddings.zip",
)
