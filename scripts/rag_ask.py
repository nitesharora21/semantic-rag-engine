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
            top_k=3)
    answer = rag.answer(question)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer)

if __name__ == "__main__":
    main()
