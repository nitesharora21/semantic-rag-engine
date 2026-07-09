# 0016 - Evaluate FAISS Retrieval Baseline

## Date

2026-07-09

## Context

Ok the FAISS-based semantic retrieval is done now - we have added the code to use FAISS,
to provide a vector search that can retrieve the nearest chunk embeddings given a
query vector and an index containing the embedding vectors.

So now we have completed 3 ways of retrieval:

1. Token based - keeping score of the tokenized keywords in query that are in the chunks
2. Cosine Similarity - taking the highest of the cosine similarity metric to get the nearest
   query vectors to the chunks
3. FAISS Vector Store - using FAISS to obtain the nearest chunk from query_vector and embedding vectors

For now just adding evaluation of the FAISS retrieval script:

```
scripts/evaluate_faiss_retrieval.py
```
