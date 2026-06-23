# 0002: Start with Basic Character - Based Chunking

## Date

2026-06-03

## Context

So the RAG engine needs the text to be chunked in smaller pieces before the embedding and retreival process.
I have added a very small code snippet to chunk the code based on the chunk size. 

Will add more modifications later to the code. 

## Decision 

The first chunking implementation will use a simple and basic character-based chunking functionality.


## Functionality

The function accepts:

- Raw text
- Chunk Size


And it returns:

- A list of str text chunks

Example:

``` python
chunk_text("abcdefghijklmno", chunk_size=3)
>> ["abc", "def", "ghi", "jkl", "mno"]
```

