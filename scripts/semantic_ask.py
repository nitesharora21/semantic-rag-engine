import sys

from rag_engine.embeddings import EmbeddingModel
from rag_engine.semantic_retriever import retrieve_semantic_chunks
from rag_engine.storage import load_chunks, load_embeddings


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/semantic_ask.py 'your question'")
        return

    query = " ".join(sys.argv[1:])

    chunks = load_chunks("data/processed/chunks.json")
    chunk_embeddings = load_embeddings("data/processed/embeddings.json")

    model = EmbeddingModel()
    query_embedding = model.embed_text(query)

    results = retrieve_semantic_chunks(
        query_embedding=query_embedding, chunks=chunks, chunk_embeddings=chunk_embeddings, top_k=3
    )

    print(f"Query: {query}")
    print(f"Showing top {len(results)} semantic matches..")

    for index, result in enumerate(results, start=1):
        score, chunk = result

        print(f"\n-- Result {index} | Similarity: {score:.4f} ---")
        print(chunk)


if __name__ == "__main__":
    main()
