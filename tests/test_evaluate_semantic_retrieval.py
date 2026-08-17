from rag_engine.retrieval_evaluation import contains_expected_terms
from rag_engine.semantic_retriever import retrieve_semantic_chunks
from rag_engine.models import Chunk


def test_semantic_retrieval_results_can_be_evaluated_with_expected_terms() -> None:
    chunks = [
        Chunk(id="doc-1:: chunk-0", document_id="doc-1", text="Nitesh knows Kafka and can build event-driven services", source="data/raw/example.txt", start_char=0, end_char=39),
        Chunk(id="doc-1:: chunk-1", document_id="doc-1", text="Nitesh studied masters in georgia tech", source="data/raw/example.txt", start_char=40, end_char=69),
        Chunk(id="doc-1:: chunk-2", document_id="doc-1", text="Developer productivity and feature development", source="data/raw/example.txt", start_char=70, end_char=126),
    ]
    chunk_embeddings = [[1.0, 0.0], [0.0, 1.0]]
    query_embedding = [1.0, 0.0]
    retrieved_chunks = retrieve_semantic_chunks(
        query_embedding=query_embedding,
        chunk_embeddings=chunk_embeddings,
        chunks=chunks,
        top_k=1,
    )
    print(f"Chunks: {chunks}, Retrieved_chunks: {retrieved_chunks}")
    result = contains_expected_terms(
        retrieved_chunks=retrieved_chunks,
        expected_terms=["Kafka", "event-driven"]
    )
    assert result is True
