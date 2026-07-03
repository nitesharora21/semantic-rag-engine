import json
from pathlib import Path

from rag_engine.retriever import retrieve_chunks
from rag_engine.evaluation import contains_expected_terms, calculate_accuracy
from rag_engine.storage import load_chunks


def load_eval_questions(file_path: str) -> list[dict]:
    """
    Loads the eval questions that are of the form:
    {
        "questions" : <question>,
        "expected_terms": [<expected_terms>]
    }
    """
    path = Path(file_path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    chunks = load_chunks("data/processed/chunks.json")
    eval_questions = load_eval_questions("eval/retrieval_questions.json")

    results_summary = []

    for item in eval_questions:
        question = item["question"]
        expected_terms = item["expected_terms"]

        retrieved_chunks = retrieve_chunks(question, chunks, top_k=3)
        success = contains_expected_terms(retrieved_chunks, expected_terms)

        # Adding success result to result_summary to calculate the accuracy score later
        results_summary.append(success)

        status = "PASSED" if success else "FAILED"
        print(f"\n[{status}] {question}")
        print(f"Expected Terms: {expected_terms}")

        for index, result in enumerate(retrieved_chunks, start=1):
            score, chunk = result
            print(f"\n Result {index} | Score: {score}")
            print(f"\tTruncated Chunk:\n{chunk}\n")
    passed_results = sum(results_summary)
    total_results = len(results_summary)
    accuracy = calculate_accuracy(results_summary)

    print("\n--- SUMMARY ---")
    print(f"Passed : {passed_results}/{total_results}")
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
