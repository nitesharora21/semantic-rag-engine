from rag_engine.evaluation import contains_expected_terms
from rag_engine.vector_store import FaissVectorStore


def test_faiss_results_can_be_evaluated_with_expected_terms() -> None:
    chunks = [
        "Nitesh has built event-driven message bus services using Kafka",
        "Nitesh has done his masters in computer science and has a patent as well",
    ]

    embeddings = [[1.0, 0.0], [0.0, 1.0]]

    store = FaissVectorStore(embeddings=embeddings)

    faiss_results = store.search(query_embedding=[1.0, 0.0], top_k=1)

    retrieved_chunks = [(score, chunks[chunk_index]) for score, chunk_index in faiss_results]

    result = contains_expected_terms(
        retrieved_chunks=retrieved_chunks, expected_terms=["Kafka", "event-driven"]
    )

    assert result is True
