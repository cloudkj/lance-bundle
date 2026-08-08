from lance_bundle import load, save, ExportDataset
from sentence_transformers import SentenceTransformer

documents = [
    "Quantum mechanics is the foundation of modern physics.",
    "Photosynthesis allows plants to convert sunlight into food.",
    "I love eating pepperoni pizza on Fridays."
]
# Row-level metadata
topics = [
    {"topic": "physics"},
    {"topic": "biology"},
    {"topic": "food"},
]

# Load model and generate embedding vectors
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(documents)

# Export bundle
output_path = "examples/01_output.zip"
save(
    model,
    ExportDataset(
        documents,
        embeddings,
        metadata=topics,
        name="sample-facts",
        description="A handful of one-line facts across physics, biology, and food.",
        source="hand-written for lance-bundle examples",
    ),
    output_path,
)

# Load bundle
bundle = load(output_path)

# Embed and query
results = bundle.search("Teach me about physics")
for result in results:
    print(f"distance={result['_distance']} document={result['text'][:30]}...")
