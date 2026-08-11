lance-bundle - Portable Embeddings
===================================

### _Embed Once, Query Forever_

[![PyPI Version](https://img.shields.io/pypi/v/lance-bundle)](https://pypi.org/project/lance-bundle/)
[![PyPI Python Version](https://img.shields.io/pypi/pyversions/lance-bundle)](https://pypi.org/project/lance-bundle/)

`lance-bundle` packages embedding vectors and the model that produced them into a single portable file - load it
and query it anywhere, with no server, no database to run, and no re-embedding your data on a new machine. Built on
LanceDB and ONNX for a minimal dependency footprint, it's equally at home embedded in an app, on the edge, or in a
local script.

## Usage

### Loading Bundles

Install: `pip install lance-bundle`

If loading bundles from Hugging Face: `pip install lance-bundle[hub]`

Import and load locally or from readily available embedding datasets ([huggingface.co/lance-bundle](https://huggingface.co/lance-bundle)):

```python
from lance_bundle import load_dataset
bundle = load_dataset("lance-bundle/berkshire-hathaway-letters")
bundle.search("What does Warren Buffett think of Coca-Cola and car insurance?")
```

### Exporting Bundles

Install: `pip install lance-bundle[export]`

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

### Local RAG Example

Add RAG to a local LLM by loading a precomputed bundled embedding dataset
([huggingface.co/lance-bundle](https://huggingface.co/lance-bundle)). Example
using Ollama:

```python
from lance_bundle import load_dataset
import ollama

bundle = load_dataset("lance-bundle/paul-graham-essays")

prompt = "You are a Silicon Valley angel investor and venture capitalist. Provide advice and answers based ONLY on the provided context. If the answer cannot be found in the context, say 'I don't know'. Context: {context} Question: {query}"

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

### CLI

```bash
# Download a bundle from the Hugging Face dataset registry
lance-bundle download lance-bundle/berkshire-hathaway-letters -o berkshire.zip

# Inspect a bundle's metadata (model, dataset, provenance)
lance-bundle inspect berkshire.zip

# Search a bundle without writing any Python
lance-bundle search berkshire.zip "What does Buffett think of Coca-Cola?" --limit 10 --json
```

## Design

### Goals

1. **Portability**: you own your data and embeddings; store and search them forever into the future
2. **Reusability**: stop re-embedding the internet; a shared dataset registry allows compressed and indexed knowledge to be used interchangeably
3. **Provenance**: each self-contained bundle contains metadata, lineage, and the exact model artifact used for generating the data

### Implementation

The initial version of `lance-bundle` is limited to support for `SentenceTransformer` embedding models and embedding
vectors stored in LanceDB. While these underlying technologies may change in the future, the aim will always be to
preserve the spirit of the portable, forwards-compatible design of `lance-bundle` and allow for supporting new and
different ways of computing, storing, and searching embeddings.

#### Sentence Transformer

By design, the Sentence Transformers models from Hugging Face are exported to ONNX with only the transformers modules. The
[pooling and normalization modules](https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/sentence_transformer/modules/pooling.py#L70-L326) are not included in the export, and thus need to be applied as a post-processing step;
`lance-bundle` ensures that the post-processing pooling and normalization steps are also included as part of the
portable package.

See: 
* https://github.com/huggingface/sentence-transformers/issues/3258
* https://sbert.net/docs/sentence_transformer/usage/efficiency.html#onnx

## License

MIT