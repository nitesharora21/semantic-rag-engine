# 0019 - Added stable chunk ID to the chunks

## Date
2026-07-11

## Context
Currently we only had chunks that were list of strings. This will be an issue
when we need to do any sort of formal analysis.
Right now any chunk containing the expected keywords is returned but if its not
relevant to user query, it still is retrieved.

Because now I added the chunk ids, I can now introduce expected chunk labels,
recall@k, mean reciprocal rank.

## Decision

So now each chunk will be of the form:
{
  "id": <unique chunk id>,
  "text": <chunk text>
}

Chunk IDs are generated in the format:
chunk-0
chunk-1
...

And the number associates to the chunk in question.

Now that we can move from expected terms to expected chunk ids, we can now
be more precise with the retrievals. Prior to this, the expected terms
would retrieve all the chunks that contain the words from query in them.
That would result in poor retrieval since the chunks might not always be
relevant.

Now - with specifying the exact chunk, we can ensure we are capturing the
correct chunk.

This will also help in scaling, as its much easier to pass through a dict,
and also helps in establishing the identity needed for future labels.

## Limitations
The chunks still dont contain more meta data around them such as:
- Source Paths
- Document IDs
- Character Offsets
- Metadata
- Content Hashes
