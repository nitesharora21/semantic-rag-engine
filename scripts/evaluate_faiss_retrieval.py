from rag_engine.embeddings import EmbeddingModel
from rag_engine.storage import load_chunks, load_embeddings
from rag_engine.vector_store import FaissVectorStore
from rag_engine.evaluation import (
    load_eval_questions,
    RetrievalResult,
    evaluate_recall_at_k,
    calculate_mean_score,
)

chunks = load_chunks("data/processed/chunks.json")
embeddings = load_embeddings("data/processed/embeddings.json")
eval_questions = load_eval_questions("eval/retrieval_questions.json")

model = EmbeddingModel()
store = FaissVectorStore(embeddings=embeddings)


def faiss_retrieve(question: str) -> list[RetrievalResult]:
    query_embedding = model.embed_text(question)
    retrieved_chunks = store.search(query_embedding=query_embedding, top_k=3)
    # Here we are doing 2 things
    # 1. Getting the chunk at index: chunk_index, and then getting the id
    # But we at the mercy that the chunk_index = "chunk-<chunk_index>"
    # This is not always true - for now this works, later we can refine
    # it even more.
    return [(score, chunks[chunk_index]["id"]) for score, chunk_index in retrieved_chunks]


def main() -> None:

    recall_scores = evaluate_recall_at_k(
        eval_questions=eval_questions, retrieve_fn=faiss_retrieve, k=3
    )
    for item, recall in zip(eval_questions, recall_scores):
        print(f"\nQuestion: {item['question']}")
        print(f"Expected Chunk IDs: {item['expected_terms']}")
        print(f"Recall@3: {recall:.2f}")
    mean_recall_score = calculate_mean_score(recall_scores)
    print("\n--- Summary ---")
    print(f"Mean Recall@3: {mean_recall_score:.2f}")


if __name__ == "__main__":
    main()
