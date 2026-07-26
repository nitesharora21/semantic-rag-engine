import sys

from rag_engine.embeddings import EmbeddingModel
from rag_engine.storage import load_chunks
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
    loaded_store = FaissVectorStore.load("data/processed/faiss.index")

    model = EmbeddingModel()
    query_embedding = model.embed_text(query)

    results = loaded_store.search(query_embedding=query_embedding, top_k=3)

    print(f"User Query: {query}")
    print(f"Showing top {len(results)} FAISS semantic matches")

    for rank, result in enumerate(results, start=1):
        score, chunk_index = result
        chunk = chunks[chunk_index]

        print(f"--- Result {rank} | Score: {score:.4f} | Chunk Index: {chunk_index} ---")
        print(f"Source: {chunk.model_dump()['source']} ")
        print(f"[{chunk.model_dump()['start_char']}: {chunk.model_dump()['end_char']}]")
        print(chunk.model_dump()['text'])


if __name__ == "__main__":
    main()

