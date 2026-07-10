from rag_engine.retriever import retrieve_chunks
from rag_engine.storage import load_chunks
from rag_engine.evaluation import (
    calculate_accuracy,
    evaluate_retriever,
    load_eval_questions,
    RetrievalResult,
)


chunks = load_chunks("data/processed/chunks.json")
eval_questions = load_eval_questions("eval/retrieval_questions.json")


def keyword_retrieve(question: str) -> list[RetrievalResult]:
    return retrieve_chunks(
        query=question,
        chunks=chunks,
        top_k=3,
    )


def main() -> None:

    results_summary = evaluate_retriever(
        eval_questions=eval_questions,
        retrieve_fn=keyword_retrieve,
    )

    for item, success in zip(eval_questions, results_summary):
        question = item["question"]
        expected_terms = item["expected_terms"]
        retrieved_chunks = keyword_retrieve(str(question))

        status = "PASS" if success else "FAIL"

        print(f"\n --- [{status}] {question}")
        print(f"Expected Terms: {expected_terms}")

        for index, result in enumerate(retrieved_chunks, start=1):
            score, chunk = result
            print(f"\n--- Result {index} | Score: {score}")
            print(f"\t{chunk[:300]}")
    passed = sum(results_summary)
    total = len(results_summary)
    accuracy = calculate_accuracy(results_summary)
    print("\n--- Summary ---")
    print(f"Passed: {passed / total}")
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
