import pytest

from rag_engine.vector_store import FaissVectorStore


def test_faiss_vector_store_can_be_saved_and_loaded(tmp_path) -> None:
    # Save FaissVectorStore embedding store in a file
    index_path = tmp_path / "test.index"
    # Sample embeddings
    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.8, 0.2]
    ]
    store = FaissVectorStore(embeddings=embeddings)
    store.save(str(index_path))
    # Then load that store using load clsmethod.
    loaded_store = FaissVectorStore.load(str(index_path))
    # Then search using the search(query_embedding, top_k) function
    results = loaded_store.search(query_embedding=[1.0, 0.0], top_k=1)
    assert index_path.exists()
    assert len(results) == 1
    # Results contain the list of tuples containing score and index, chose the first one
    score, index = results[0]
    assert index == 0
    assert score > 0.99

def test_loaded_faiss_index_preserves_dimension(tmp_path) -> None:
    index_path = tmp_path / "test.index"
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ]
    store = FaissVectorStore(embeddings=embeddings)
    store.save(index_path)
    loaded_store = FaissVectorStore.load(str(index_path))
    assert loaded_store.dimension == 3


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
