def retreive_chunks(query: str, chunks: list[str]) -> list[str]:
    """
    So this is the most basic version of retreival, which is
    if the word if found in the chunk - return that chunk.

    Keeping it super simple for now.
    """
    query_words = query.lower().split()
    results = []

    for chunk in chunks:
        chunk_lower = chunk.lower()
        for word in query_words:
            if word in chunk_lower:
                results.append(chunk)
                break
    return results
