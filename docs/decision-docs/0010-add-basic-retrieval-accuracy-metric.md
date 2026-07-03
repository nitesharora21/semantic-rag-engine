# 0010: Added basic retrieval accuracy metric scores (pre-cursor to ranking)

## Date
2026-07-03

## Context

Ive added a small retrieval evaluation dataset to the project.

This is the first step of creating an evaluation loop. 

Each evaluation item contains a user question and the expected terms that should 
appear in the retrieved chunks - retrieved using word count of query terms inside the processed data chunks. 

Once the evaluation provides a bool after checking whether the expected_terms
were in the retrieved_chunks or not, the results are tracked in results summary
list. 

Finally the accuracy is calculated by checking the total number of SUCCESS
retrieved_chunks / total number of checks done, i.e. # of SUCCESS expected terms in retrieved_chunks given query / total # of queries

So the basic metric is as follows:
```
accuracy = number of passed evaluation questions / total # of evaluation questions
```