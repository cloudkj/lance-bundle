lance-bundle -  Portable Embeddings
===================================

### _Embed Once, Query Forever_

`lance-bundle` is a library and opinionated packaging format for portable, self-contained bundles of embedding vectors
and models for local-first, forwards-compatible use across AI and information retrieval.

The process of loading and querying embeddings is built on top of LanceDB and the ONNX runtime, and designed to be
lightweight in terms of dependencies to allow for use in embedded, edge, and local environments.

## Quick Start

Installation: `pip install lance-bundle`

Import readily available embedding datasets from HuggingFace ([huggingface.co/lance-bundle](https://huggingface.co/lance-bundle)):

```python
from lance_bundle import load_dataset
bundle = load_dataset("lance-bundle/berkshire-hathaway-letters")
bundle.search("What does Warren Buffett think of Coca-Cola and car insurance?")
```

Export documents and embeddings to a local archive file (zip), then load as needed for queries:

```python
from lance_bundle import load, save, ExportDataset
from sentence_transformers import SentenceTransformer
documents = [
    "Quantum mechanics is the foundation of modern physics.",
    "Photosynthesis allows plants to convert sunlight into food.",
    "I love eating pepperoni pizza on Fridays.",
]
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
save(model, ExportDataset(texts=documents, vectors=model.encode(documents)), "bundle.zip")
# ...
bundle = load("bundle.zip")
bundle.search("I'm hungry for some science")
```

## Goals

1. Portability: you own your data and embeddings; store and search them forever into the future
2. Reusability: stop re-embedding the internet; a shared dataset registry allows compressed and indexed knowledge to be used interchangeably
3. Provenance: each self-contained bundle contains metadata, lineage, and the exact model artifact used for generating the data

## Examples

Add RAG to a local LLM by loading a precomputed bundled embedding dataset from HuggingFace
([huggingface.co/lance-bundle](https://huggingface.co/lance-bundle)):

```python
from lance_bundle import load_dataset
import ollama

bundle = load_dataset("lance-bundle/paul-graham-essays")

prompt = "You are a Silicon Valley angel investor and ventural capitalist. Provide advice and answers based ONLY on the provided context. If the answer cannot be found in the context, say 'I don't know'. Context: {context} Question: {query}"

while True:
    print("> ", end="")
    query = input()
    context = bundle.search(query, limit=1)[0]['text']
    response = ollama.chat(
        model='llama3.2:1b',
        messages=[{'role': 'user', 'content': prompt.format(context=context, query=query)}]
    )
    print(response.message.content)
```

## CLI

```bash
# Download a bundle from the HuggingFace dataset registry
lance-bundle download lance-bundle/berkshire-hathaway-letters -o berkshire.zip

# Inspect a bundle's metadata (model, dataset, provenance)
lance-bundle inspect berkshire.zip

# Search a bundle without writing any Python
lance-bundle search berkshire.zip "What does Buffett think of Coca-Cola?" --limit 10 --json
```

## Implementation

The initial version of `lance-bundle` is limited to support for `SentenceTransformer` embedding models and embedding
vectors stored in LanceDB. While these underlying technologies may change in the future, the aim will always be to
preserve the spirit of the portable, forwards-compatible design of `lance-bundle` and allow for supporting new and
different ways of computing, storing, and searching embedding spaces indefinitely into the future.

#### Sentence Transformer

By design, the Sentence Transformers models from HF are exported to ONNX with only the transformers modules. The
[pooling and normalization modules](https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/sentence_transformer/modules/pooling.py#L70-L326) are not included in the export, and thus need to be applied as a post-processing step.
This is an additional benefit to using this library to import an ONNX-exported sentence transformer for embeddings as
that post-processing step also gets included into the bundle.

See: 
* https://github.com/huggingface/sentence-transformers/issues/3258
* https://sbert.net/docs/sentence_transformer/usage/efficiency.html#onnx

## License

MIT