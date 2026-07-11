from rag_engine.embeddings import EmbeddingModel
from rag_engine.storage import load_chunks, save_embeddings


def main() -> None:
    chunks = load_chunks("data/processed/chunks.json")
    print(f"Loaded Chunks: {chunks}")
    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddingModel = EmbeddingModel()
    embeddings = embeddingModel.embed_texts(chunk_texts)
    save_embeddings(embeddings, "data/processed/embeddings.json")
    print(f"Length of loaded chunks: {len(chunks)}")
    print(f"Saved {len(embeddings)} embeddings into data/processed/embeddings.json")
    if embeddings:
        print(f"Embedding dimension (each vector): {len(embeddings[0])}")


if __name__ == "__main__":
    main()
