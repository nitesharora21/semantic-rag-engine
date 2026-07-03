import re


def tokenize(text: str) -> list[str]:
    """
    Identifies words within the \b boundary, which is a good idea to
    get the words.
    Without tokenizing the text, substrings as small as a single character will
    be included in the scoring and that is not correct.
    """
    return re.findall(r"\b\w+\b", text.lower())


def score_chunk(query: str, chunk: str) -> int:
    """
    Calculate the number of words from the query in the chunk.
    We need that to know how much score to give to the query - higher if more
    words match in the chunk.
    """
    score = 0
    chunk_lower = tokenize(query)
    query_lower = tokenize(chunk)

    if isinstance(query_lower, str):
        query_lower = [query_lower]

    for query_word in query_lower:
        if query_word in chunk_lower:
            score += chunk_lower.count(query_word)

    return score


def retrieve_chunks(query: str, chunks: list[str], top_k: int = 3) -> list[str]:
    """
    So this is the most basic version of retreival, which is
    if the word if found in the chunk - return that chunk.
    Ok now adding the scoring method - score how much of a query
    is against a chunk - then rank the highest scoring chunks. Then
    return the chunks and the score with it as well.
    """
    scored_chunks = []
    for chunk in chunks:
        score = score_chunk(query, chunk)
        if score > 0:
            # Putting score first cause the sorted will use the score to sort instead of chunk
            scored_chunks.append((score, chunk))
    # Need to sort by highest score first
    scored_chunks.sort(reverse=True)
    return scored_chunks[:top_k]
