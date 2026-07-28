import sys

from rag_engine.embeddings import EmbeddingModel
from rag_engine.prompting import build_rag_prompt
from rag_engine.storage import load_chunks
from rag_engine.vector_store import FaissVectorStore

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python script/build_prompt.py 'your question' ")
        return 

    question = " ".join(sys.argv[1:])
    
    chunks = load_chunks("data/processed/chunks.json")
    # Fiass Index is already created from the chunk embeddings, so no need for chunk embeddings anymore
    store = FaissVectorStore.load("data/processed/faiss.index")
    
    model = EmbeddingModel()
    query_embedding = model.embed_text(question)
    
    # Returns the score and the chunk index
    results = store.search(query_embedding=query_embedding, top_k=3)

    # With the score and chunk_index, get the relevant chunks
    retrieved_chunks = [chunks[chunk_index] for score, chunk_index in results]

    # Pass in the user query and the relevant chunk text into the build_rag_prompt
    prompt = build_rag_prompt(question=question, chunks=retrieved_chunks)

    # For now print prompt, will add the generic LLM model to feel this data into
    print(prompt)

if __name__ == "__main__":
    main()
