import json
from pathlib import Path

from rag_engine.retriever import retrieve_chunks
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


def contains_expected_terms(retrieved_chunks: list[str], expected_terms: list[str]) -> bool:
    """
    Takes in the retrieved_chunks based on the question asked.
    Then the expectation is that the chunks will have those expected terms in it.
    If the chunks dont have those expected_terms, then return False, else return True.
    """
    combined_chunks = " ".join(chunk for _, chunk in retrieved_chunks).lower()
    # All the terms are supposed to be inside the combined_chunks, if even a single is missing return False
    for term in expected_terms:
        if term.lower() not in combined_chunks:
            return False
    return True


def main() -> None:
    chunks = load_chunks("data/processed/chunks.json")
    eval_questions = load_eval_questions("eval/retrieval_questions.json")

    passed = 0

    for item in eval_questions:
        question = item["question"]
        expected_terms = item["expected_terms"]

        results = retrieve_chunks(question, chunks, top_k=3)
        success = contains_expected_terms(results, expected_terms)
        if success:
            passed += 1
        status = "PASSED" if success else "FAILED"
        print(f"\n[{status}] {question}")
        print(f"Expected Terms: {expected_terms}")

        for index, result in enumerate(results, start=1):
            score, chunk = result
            print(f"\n Result {index} | Score: {score}")
            print(f"\tTruncated Chunk:\n{chunk}\n")
    total_questions = len(eval_questions)
    # Basically how many retrieved chunks per query has the expected terms in it
    accuracy = passed / total_questions if total_questions else 0

    print("\n--- SUMMARY ---")
    print(f"Passed : {passed}/{total_questions}")
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
