# 0013: Added Cosine Similarity Semantic Retrieval

## Date:
2026-07-05

## Context:

Ok now the project can do cosine similarity between 2 vectors and provide the score which is 
the cosine similarity metric. With this we can now take the query, embed into an embedding,
generate an embedding vector from it, query it against all the existing embeddings from the
processed chunks before and find the cosine similarity.
The similar embeddings will score high, and we will show the score with the output as well.

For now, its working but will require a lot more enhancements and also will need to run quicker.

Next, I will use the same test to identify the accuracy of the input user queries, with
successful chunks that contain the expected terms against total number of queries being asked.

Once I have code structure done, I will add FAISS vector search.


