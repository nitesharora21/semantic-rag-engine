import pytest

from rag_engine.vector_store import FaissVectorStore


def test_faiss_vector_store_returns_nearest_vector() -> None:
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]]
    store = FaissVectorStore(embeddings=embeddings)
    results = store.search(query_embedding=[1.0, 0.0], top_k=1)

    assert len(results) == 1

    score, index = results[0]

    assert index == 0
    assert score > 0.99


def test_faiss_vector_store_rejects_empty_embeddings() -> None:
    with pytest.raises(ValueError, match="embeddings"):
        FaissVectorStore([])
