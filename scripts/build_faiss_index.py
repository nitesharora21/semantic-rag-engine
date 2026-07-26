from rag_engine.storage import load_embeddings
from rag_engine.vector_store import FaissVectorStore

def main() -> None:
  embeddings = load_embeddings("data/processed/embeddings.json")
  store = FaissVectorStore(embeddings=embeddings)
  store.save("data/processed/faiss.index")
  print(f"Indexed {len(embeddings)} embeddings")
  print("Saved FAISS Index to data/processed/faiss.index")

if __name__ == "__main__":
  main()
