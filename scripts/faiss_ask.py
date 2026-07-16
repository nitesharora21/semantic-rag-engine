import sys

from rag_engine.embeddings import EmbeddingModel
from rag_engine.storage import load_chunks, load_embeddings
from rag_engine.vector_store import FaissVectorStore


def main() -> None:
    """
    User prompt - convert to embedding -> store chunked embedding in FAISS -> search index in FAISS
    """
    if len(sys.argv) < 2:
        print("Usage: python scripts/faiss_ask.py 'your question'")
        return

    query = " ".join(sys.argv[1:])

    chunks = load_chunks("data/processed/chunks.json")
    embeddings = load_embeddings("data/processed/embeddings.json")

    model = EmbeddingModel()
    query_embedding = model.embed_text(query)

    store = FaissVectorStore(embeddings=embeddings)
    results = store.search(query_embedding=query_embedding, top_k=3)

    print(f"User Query: {query}")
    print(f"Showing top {len(results)} FAISS semantic matches")

    for rank, result in enumerate(results, start=1):
        score, chunk_index = result
        chunk = chunks[chunk_index]

        print(f"--- Result {rank} | Score: {score:.4f} | Chunk Index: {chunk_index} ---")
        print(f"Source: {chunk['source']} ")
        print(f"[{chunk['start_char']}: {chunk['end_char']}]")
        print(chunk['text'])


if __name__ == "__main__":
    main()

