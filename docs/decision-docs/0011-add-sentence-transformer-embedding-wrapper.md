# 0011: Added sentence transformer embedding wrapper

## Date:
2026-07-04

## Context:
Switching gears to a proper retrieval system. Before I was collecting the tokenized keywords and matching them from the query with the chunks.
That was fine for baseline implementation - now we need a proper way of retrieval.

In this case - we are now embedding the input text (query, chunks, etc) to vectors using sentence_transformers library using the all-MiniLM-L6-v2 model.

This will give us a lot more accurate and precise results as compared to the simple keyword matching and scoring.

For now the basic embedding is done - will integrate in the next steps.rrag-overview.md

## Decision

For now - the project will use the sentence_transformers as the first embedding library. This will be later changed.

Embedding model
```
all-MiniLM-L6-v2
```
