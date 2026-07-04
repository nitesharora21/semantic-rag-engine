# 0012: Persist Chunk Embeddings as JSON

## Date
2026-07-04

## Context

The wrapper is ready - the chunks are converted into embeddings - so text to numeric vectors.

Now we need to generate embeddings for the persisted text chunks. These embeddings will later be used for semantic search.

Right now I dont need the vector database - or FAISS index. Will add that later. 

For now - just need to ensure that the text can be converted into an embedding and can be used later on.

## Decision

The project will add a simple script that reads chunks from:

```
data/processed/chunks.json 
```

and then generates the embeddings using the provided sentence-transformer model and saves them to:
```
data/processed/embeddings.json 
```

