import sys

from rag_engine.embeddings import EmbeddingModel
from rag_engine.generator import OllamaGenerator
from rag_engine.rag import RAGEngine

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/rag_ask.py 'your question' ")
        return
    question = " ".join(sys.argv[1:])

    rag = RAGEngine(
            chunks_path="data/processed/chunks.json",
            index_path="data/processed/faiss.index",
            embedding_model=EmbeddingModel(),
            generator=OllamaGenerator(model_name="gemma3"),
            top_k=3,
            minimum_score=0.5)
    response = rag.answer(question)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(response.answer)

    print(f"\nAbstained: {response.abstained}")
    print("\nSources:")
    for rank, result in enumerate(response.sources, start=1):
        if result.score < rag.minimum_score:
            continue
        chunk = result.chunk
        print(f"\n{rank}. "
              f"{chunk.id}"
              f"(score={result.score:4f})")
        print(f"\t{chunk.source}"
              f"[{chunk.start_char}:{chunk.end_char}]")


if __name__ == "__main__":
    main()
