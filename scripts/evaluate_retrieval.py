from rag_engine.retriever import retrieve_chunks
from rag_engine.storage import load_chunks
from rag_engine.evaluation import (
    evaluate_retriever,
    load_eval_questions,
    RetrievalResult,
)


chunks = load_chunks("data/processed/chunks.json")
eval_questions = load_eval_questions("eval/retrieval_questions.json")
chunk_texts = [chunk["text"] for chunk in chunks]
chunk_id_by_text = {chunk["text"]: chunk["id"] for chunk in chunks}


def keyword_retrieve(question: str) -> list[RetrievalResult]:
    results = retrieve_chunks(query=question, chunks=chunks, top_k=3)
    return [(score, chunk_id_by_text[chunk_text]) for score, chunk_text in results]


def main() -> None:

    results_summary = evaluate_retriever(
        eval_questions=eval_questions,
        retrieve_fn=keyword_retrieve,
    )

    recall_scores = evaluate_recall_at_k(
        eval_questions=eval_questions, retrieve_fn=keyword_retrieve, k=3
    )
    for item, recall in zip(eval_questions, recall_scores):
        print(f"\nQuestion: {item['question']}")
        print(f"Expected chunks: {item['expected_chunk_ids']}")
        print(f"Recall@3: {recall:.2f}")
    mean_recall = calculate_mean_score(recall_scores)
    print("\n---Summary ---")
    print(f"Mean Recall@3: {mean_recall:.2f}")


if __name__ == "__main__":
    main()
