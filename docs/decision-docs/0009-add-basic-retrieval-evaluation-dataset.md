# 0009: Add Basic Retrieval Evaluation Dataset

## Date

2026-07-02

## Context
The project now has a simple keyword based retrieval with scoring and tokenization of query.

Im going to tackle evaluation next, the project needs a small repeatable way to check whether retrieval is
returning useful chunks or not.

## Decision

So this project will add a small retrieval evaluation dataset at:

``` text
eval/retrieval_questions.json
```
