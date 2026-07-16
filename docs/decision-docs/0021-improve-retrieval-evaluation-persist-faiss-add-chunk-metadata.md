# 0021: Improved Retrieval Evaluation, Persisted FAISS Index and Added Meta Data to Chunks

## Date
2026-07-19

## Context
### Recall@K and MRR
Ok I have added the Recall@K which measures whether relevant chunks appeared in top K chunks or not.
So that was the first evaluation metric I added, but also I need to know the order of the chunks, i.e.
which chunks are most relevant. In order to deal with that, I added MRR - Mean Reciprocal Rank.

MRR - helps in ordering the chunks based on their ranks which computed as follows:
First relevant chunk at rank 1: 1 / 1 = 1.00
First relevant chunk at rank 2: 1 / 2 = 0.50
First relevant chunk at rank 3: 1 / 3 = 0.33
If no relevant chunk retrieved:         0.00

And MRR - is the average reciprocal rank across all evaluation questions.

So now, we have the Mean Recall@K, and Mean Reciprocal Rank.

### FAISS Index Persistance 
And with FAISS Index - since we were calling the FaissVectorStore(embeddings), we were creating
the index from scratch again and again.

Technically the index should belong to the ingestion and chunking and not query prompts.
Each time a query prompt is requested, it shouldnt generate the index again.
So we avoid rebuilding indexes, reduce repeated computation, makes query scripts simpler and allows scaling amongst many other benefits.

### Chunk Source Metadata and character offsets
Ive added some metadata to each chunk such as id, document_id, source, start_char, end_char.
I added this for a few reasons:
1. Need to enable citations - this will be needed to cite where the chunks are coming from when receiving answer from open-source language model
2. Once the dataset gets large, I need to debug it and for that I need the metadata to immediately identify the chunks
3. Verifying the sources - if the chunk retrieved is from a non-relevant source, I can see that quickly. It actually is a part of step 2 but still making the point.

### Note:
Recall@K - How many relevant chunks appeared in top-k results?
MRR - How soon did the first relevant chunk appeared?

And this ranked relevant chunks are very important as they will guide the downstream prompt
and influence the generated answer.

For FAISS Vector Index: Just need to remember that if any of the source document changes - the entire step of ingesting, chunking, creating index and storing has to happen again.

So the flow might now look like:
retrieved chunks
    |
    V
prompt construction
    |
    V
open-source language model
    |
    V
grounded answer
    |
    V
citations

## Tradeoffs
1. Now since the FAISS Index is a seperate entity, we need to make sure that the ingestion, embedding and index creation happen at the same time, otherwise there will sync issues
2. The retrieval metrics are evaluated on the manually labelled chunks. With any change in the document, labelling has to be done, due to character offsets shifting
3. Once the dataset is scaled to a large amount, more validations are needed - will discuss that in the coming changes.

