# 0006: Persist Chunks as JSON

## Date

2026-06-24

## Context

So the ingestion script now loads the raw document, splits it into chunks and now stores the chunks in a seperate file.

The processed chunked file is now persisted in 

```data/processed/chunks.json```
