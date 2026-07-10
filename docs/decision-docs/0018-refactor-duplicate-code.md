# 0018 Removed duplicated code for retrieval evaluation helpers

## Date
2026-07-10

## Context
 There was quite a lot of duplicate code in the scripts.
 All the retrieval functions for all different types of retrieval
 were called - however they were mostly the same.

 We now have multiple retrieval evaluation scripts:
 - Keyword retrieval evaluation
 - Manual semantic retrieval evaluation
 - FAISS retrieval evaluation
And also the comparison of all those retriever methods.

Now that the duplicate logic is fixed, changes will be more precise and put together
easily.

One of the examples such as recall@k or mean reciprocal rank will effect
all the methods, and now I don't have to make changes in all seperate scripts.


