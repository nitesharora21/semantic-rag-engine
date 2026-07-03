from rag_engine.embeddings import EmbeddingModel


def test_embedding_text_returns_vector() -> None:
    model = EmbeddingModel()

    embedding = model.embed_text("Im am building a RAG project from scratch")

    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert isinstance(embedding[0], float)
