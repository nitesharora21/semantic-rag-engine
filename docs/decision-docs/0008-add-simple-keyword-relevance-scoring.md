# 0008: Add Simple Keyword Relevant Scoring

## Date

2026-06-26

## Context

So now instead of returning a chunk that has the word from the query - we now calculate the instance of each word
from the query and use that as a score to rank the chunk.

Now with that being said, there is no reason to believe that higher number of words in a chunk means higher similarity.
Its just one factor, with lots of other factors that define the similarity of the prompt with the chunks.

## Example

Given the query asked from ask.py
```
python scripts/ask.py "python developer
```
Following is what we see in the output:
```
'ples include:\n\n* Migrating from a legacy proprietary source control system to Git\n* Supporting multi-site mirror infrastructure across multiple geographic regions\n* Migrating a platform from Python 2 to Python 3\n* Moving from batch-based resource scheduling to on-demand queued build scheduling\n\nThes'
```
