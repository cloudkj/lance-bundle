lance-bundle
============

### Motto: Embed Once, Query Forever

Main concrete goals:
1. Portability - you own your data and embeddings; store them and use them forever into the future
2. Reusability - stop embedding the internet; a shared registry allows compressed and indexed knowledge to be used interchangeably
3. Provenance - metadata and lineage information to track snapshots and versioning and enable time-travel, debugging, and auditing

Other taglines:
* "stop rembedding the wheel"

## Motivating Examples

Add RAG to local LLM with one-liner to load bundle as dataset from Hugging Face:

```python
import ollama
from lance_bundle.runtime import load_dataset

bundle = load_dataset("cloudkj/lance-bundle-mvp")

prompt = "You are a helpful assistant. Answer the question based ONLY on the provided context. If the answer cannot be found in the context, say 'I don't know'. Context: {context} Question: {query}"

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

## Implementation Notes

By design, the Sentence Transformers models from HF are exported to ONNX with only the transformers modules. The
pooling and normalization modules are not included in the export, and thus need to be applied as a post-processing step.
This is an additional benefit to using this library to import an ONNX-exported sentence transformer for embeddings.

Source pointer for pooling logic:
* https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/sentence_transformer/modules/pooling.py#L70-L326

Discussions:
* https://github.com/huggingface/sentence-transformers/issues/3258
* https://sbert.net/docs/sentence_transformer/usage/efficiency.html#onnx

## Datasets

Some datasets for which we can generate embeddings to bootstrap registry:
* https://ai.google.com/research/NaturalQuestions

## Ideas

* Vector DB as analytics/OLAP?
  * Support operations like "how many documents are related to topic X?" - requires an operation more like
    `SELECT count(1) WHERE distance(document, query) < threshold`
  * Or clustering operations? ooks like Inverted File Index (IVF) does some clustering
  * What about pure aggregations like "what percent of my reviews were negative?"
* Value-adds:
  * snapshot and bundle cersioning, history, provenance. need very detailed audit trail of how snaoshot waa produced to ensure reliable drozen snapshot in time and allow dor rollbacks and time travel ans introspection during tegressions ir changes in behavior 
  * value add with APi on top: add anayltics aggregation funtionality in top of cector db for more powerful searches, aggregations, etc
