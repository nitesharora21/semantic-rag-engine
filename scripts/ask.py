import sys

from rag_engine.retriever import retrieve_chunks
from rag_engine.storage import load_chunks


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/ask.py 'your question'")
        return
    query = " ".join(sys.argv[1:])
    chunks = load_chunks("data/processed/chunks.json")
    results = retrieve_chunks(query, chunks, top_k=3)

    print(f"Query: {query}")
    print(f"Results: {results}")

    for index, chunk in enumerate(results, start=1):
        print(f"\n---Result: {index}---")
        print(chunk)


if __name__ == "__main__":
    main()
