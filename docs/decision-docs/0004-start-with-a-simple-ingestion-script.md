# 0004: Start with a Simple Ingestion Script

## Date

2026-06-23

## Context

The project now has two basic building blocks:

- A local text file loader
- A simple character-based chunker

The next step is to connect these pieces into a minimal ingestion flow.

At this stage, the project does not need a full pipeline abstraction, configuration system, vector store, embeddings, or persistence.

## Decision

The first ingestion script will:

1. Load one sample text file from `data/raw/`
	Im thinking of adding my resume to the project as the input document - will be cool to ask questions regarding my resume.
2. Split the text into chunks
3. Print the chunks to the terminal

The script will live at:

```text
scripts/ingest.py
```
