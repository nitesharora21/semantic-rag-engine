# 0015: Added FAISS Vector Search

## Date

2026-07-08

## Context

Ok so now we have added the FAISS vector retrieval - as we need to have a vector search
mechanism that can retrieve nearest embeddings efficiently.

Prior to this - we have already added the code to retrieve the embeddings using the cosine
similarity which is great to understand the underlying concepts - but with in mind,
we need to be able to scale the project as well, hence FAISS.

## Decision

For now I have added the FAISS (Facebook AI Semantic Similarity) Vectore Store

The flow is the same as other mechanisms (e.g. token based, cosine simiarity based):

1. Load chunk embeddings from `embeddings.json`
2. Convert those embeddings into NumPy Array
3. Normalize embeddings
4. Build an in-memory FAISS index
5. Search the index using a normalized query embedding (note: reshape to fit the index dimension)
6. Return similarity scores and chunk indices

The vector store will live in:

```
src/rag_engine/vector_store.py
```
