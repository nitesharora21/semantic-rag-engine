# A semantic RAG Engine

Building a RAG based system that will retrieve the most relevant
document chunks from a knowledge base and uses those chunks
as grounded context to generate an answer with supporting sources. 

Traditional keywords searches often fail or worse hallucinate, especially 
when the users ask questions in different wording than what the documents
use. The current gen LLM models can generate fluent answers, but without
the proper semantic similarity between the users questions and the documents
the LLM models can hallucinate or provide stale answers.

The goal here is to develop the project in 3 main phases:
1. Baseline RAG: Setup a baseline RAG architecture, load the documents, chunk the text, add embedding model, etc. until we are at the command-line question answering
2. Evaluation: TBA
3. Production Features: TBA


## Planned Milestone (Will add more as we go)

### Phase 1: Baseline RAG
- [ ] Add document loader
- [ ] Add text chunking
- [ ] Add embedding model
- [ ] Add vector store
- [ ] Add retriever
- [ ] Add answer generator
- [ ] Add command-line question answering

### Phase 2: Evaluation
- [ ] TBA

### Phase 3: Production Features
- [ ] TBA


## Project Structure

``` text
semantic-rag-engine/
  src/                  # This is where the main src code lies for rag_engine
    rag_engine/
      config.py
      loaders.py
      chunking.py
      embeddings.py
      vector_store.py
      retriever.py
      generator.py
      pipeline.py
  data/                 # All the data will reside here, raw and processed both
    raw/
    processed/
  scripts/              # Additional helper scripts that will help in ingesting and asking questions via prompt.
    ingest.py
    ask.py
  tests/                # Tests - important, well when are they not important?
```


