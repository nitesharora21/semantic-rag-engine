# 0017 Added comparisons of retrieval methods

## Date
2026-07-09

## Context

Added code to compare the 3 different retrieval methods:
1. Keyword Retrieval
2. Manual semantic retrieval (Cosine Similarity)
3. FAISS Semantic Retrieval

Right now, the set of eval questions contains very few items - which is why
the overall scores for all the methods is same. Out out 3 questions, each of them
provided the chunks that contained the expected terms for 2 questions.

Overtime, I will add more questions to the eval questions list and will compare
the scoring again.

## Decision

Adding the comparison script called:

```
scripts/compare_retrieval_methods.py
```

