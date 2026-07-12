# 0020: Added recall@k for retriever evaluation
## Date:
2023-07-12
## Context
So now we have added the recall@k, which helps us calculate the score of our retrieval chunks.
Now that we have enabled retrieval using exact chunk ids that we need, we can now use recall@k.
recall@k tells us what chunks retrieved contain the chunks that are expected from the expected list.
With this - we have now enabled a proper, formal, retrieval metric based on those chunk labels.
## Decision
So now the evaluation dataset will now show the chunk ids instead of the keywords that we are looking for.
This is a better option since just blindingly checking the expected terms, will retrieve non-semantically
similar chunks.
Now the processed chunks look like:
```
{
  "question": "What experience does Nitesh have with Machine Learning?"
  "expected_terms": ["chunk-4", "chunk-12"]
}
```
And with this implementation above, we can now use recall@k, which is defined as:
$$
\text{Recall@k} = \frac{\text{Number of Relevant Chunks Retrieved in Top } k}{\text{Total Number of Relevant Chunks}}
$$
## Limitations
In this case, we are expecting the chunk id to be same as the index of the chunk. So chunk-0 must mean that the chunk resides at index 0.
And at any case, if we are modifying the documents, adding or deleting anything from the raw data, we need to re-do the entire ingestion again, ensuring the chunks-<id> is matching the index of the item in the list.I will fix that in the later commits, for now, this works fine.

Additionally, Im not reading into the order of the chunks retrieved. I.e. the first retrieved chunk has the same priority of the last retrieved chunk.
