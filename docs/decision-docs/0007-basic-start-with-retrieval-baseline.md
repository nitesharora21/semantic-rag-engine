# 0007: Start with basic Retrieval Baseline code

## Date:

2026-06-25

## Context:

Ok so now the project can read a text file, chunk it and store it in a predefined location
with a predefined name. 

Then the retrieval which retrieves based on the keyword match, retrieves the chunk whenever
there is a word match. 

Now the full fledged RAG system will use the vector embeddings and vector search for 
semantically similar retrieval of chunks. 

That will be coming next - for now the most basic version of retrieval is done. 

## Decision

For now the most basic retrieval is just keyword matching in the chunk. 

So all what it does is:
1. Takes the query, converts it into lower case.
2. Loads the processed chunks from the processed file, then
3. Iterates over the chunks and if any of the query word is in the chunk
4. Then append that chunk in the result and just show those chunks
5. In case there is no match - then the results are empty list and the pytests for ask will fail


Very very simple - but baseline is set. Incremental updates will help see how
this project slowly builds on the current RAG model.


