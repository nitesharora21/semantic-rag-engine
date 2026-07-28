from rag_engine.semantic_retriever import cosine_similarity, retrieve_semantic_chunks
from rag_engine.models import Chunk


def test_cosine_similarity_returns_one_for_same_direction() -> None:
    vector_a = [1.0, 0.0]
    vector_b = [1.0, 0.0]
    similarity = cosine_similarity(vector_a, vector_b)
    assert similarity == 1.0


def test_retrieve_semantic_chunks_returns_best_matches_first() -> None:
    chunks = [
        Chunk(id="doc-1:: chunk-0", document_id="doc-1", text="Kafka event-driven platform work", source="data/raw/example.txt", start_char=0, end_char=39),
        Chunk(id="doc-1:: chunk-1", document_id="doc-1", text="Working in cisco for a long time", source="data/raw/example.txt", start_char=40, end_char=69),
        Chunk(id="doc-1:: chunk-2", document_id="doc-1", text="Developer productivity and feature development", source="data/raw/example.txt", start_char=70, end_char=126),
    ]
    chunk_embeddings = [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]]
    query_embedding = [1.0, 0.0]
    results = retrieve_semantic_chunks(
        query_embedding=query_embedding,
        chunk_embeddings=chunk_embeddings,
        chunks=chunks,
        top_k=2
    )
    assert results[0] == (1.0, "Kafka event-driven platform work")
    assert len(results) == 2
