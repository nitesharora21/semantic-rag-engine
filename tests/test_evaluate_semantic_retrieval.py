from rag_engine.evaluation import contains_expected_terms
from rag_engine.semantic_retriever import retrieve_semantic_chunks


def test_semantic_retrieval_results_can_be_evaluated_with_expected_terms() -> None:
    chunks = [
        {"id": "chunk-0", "text": "Nitesh knows Kafka and can build event-driven services"},
        {"id": "chunk-1", "text": "Nitesh studied masters in georgia tech"},
    ]
    chunk_embeddings = [[1.0, 0.0], [0.0, 1.0]]
    query_embedding = [1.0, 0.0]
    retrieved_chunks = retrieve_semantic_chunks(
        query_embedding=query_embedding, chunks=chunks, chunk_embeddings=chunk_embeddings, top_k=1
    )
    print(f"Chunks: {chunks}, Retrieved_chunks: {retrieved_chunks}")
    result = contains_expected_terms(
        retrieved_chunks=retrieved_chunks, expected_terms=["Kafka", "event-driven"]
    )
    assert result is True
