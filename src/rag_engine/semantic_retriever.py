import math
from rag_engine.evaluation import RetrievalResult


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """
    Calculate the cosine similarity between two vectors.
    Cosine similarity measures whether the two vectors are in same direction.
    Higher values mean more in the same direction.
    """
    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for value_a, value_b in zip(vector_a, vector_b):
        dot_product += value_a * value_b
        norm_a += value_a * value_a
        norm_b += value_b * value_b

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (math.sqrt(norm_a) * math.sqrt(norm_b))


def retrieve_semantic_chunks(
    query_embedding: list[float],
    chunks: list[str],
    chunk_embeddings: list[list[float]],
    top_k: int = 3,
) -> list[RetrievalResult]:
    """
    Retrieve chunks ranked by cosine similarity to the query embedding.

    Returns:
        A list of tuples in the format:
            (similarity_score, chunk)
    """
    scored_chunks = []

    for chunk, chunk_embedding in zip(chunks, chunk_embeddings):
        score = cosine_similarity(query_embedding, chunk_embedding)
        scored_chunks.append((score, chunk))
    scored_chunks.sort(reverse=True)

    return scored_chunks[:top_k]
