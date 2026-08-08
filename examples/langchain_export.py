from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import requests
from lance_bundle import save, ExportDataset

url = "https://raw.githubusercontent.com/hwchase17/chat-your-data/master/state_of_the_union.txt"
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50,
    length_function=len,
)
documents = text_splitter.create_documents([requests.get(url).text])

print(f"Loaded documents {len(documents)=}")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print(f"HF model card {embedding_model.client.model_card_data}")

vector_store = InMemoryVectorStore(embedding_model)
vector_store.add_documents(documents=documents)

doc_ids = [doc_id for doc_id, _ in vector_store.store.items()]
texts = [item['text'] for _, item in vector_store.store.items()]
embeddings = [item['vector'] for _, item in vector_store.store.items()]

print(f"{len(texts)=}, {len(embeddings)=}")

# Export
save(
    embedding_model.client,
    ExportDataset(
        texts,
        embeddings,
        metadata=[{"doc_id": doc_id} for doc_id in doc_ids],
        description="lance-bundle LangChain export example",
    ),
    "examples/state_of_the_union_embeddings.zip",
)
