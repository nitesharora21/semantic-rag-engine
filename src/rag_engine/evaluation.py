import json
from pathlib import Path
from typing import Callable

RetrievalResult = tuple[int | float, str]  # Generally score and chunk
EvalQuestion = dict[str, list[str]]  # Question and the expected terms (str or list)


def load_eval_questions(file_path: str) -> list[EvalQuestion]:
    """
    Load the evaluation questions with their expected terms and return as a json dict"""
    path = Path(file_path)
    return json.loads(path.read_text(encoding="utf-8"))


def contains_expected_terms(
    retrieved_chunks: list[RetrievalResult], expected_terms: list[str]
) -> bool:
    """
    Takes in the retrieved_chunks based on the question asked.
    Then the expectation is that the chunks will have those expected terms in it.
    If the chunks dont have those expected_terms, then return False, else return True.
    """
    combined_chunks = " ".join(chunk for _, chunk in retrieved_chunks).lower()
    # All the terms are supposed to be inside the combined_chunks, if even a
    # single is missing return False
    for term in expected_terms:
        if term.lower() not in combined_chunks:
            return False
    return True


def calculate_accuracy(results: list[bool]) -> float:
    """
    Calculate the fraction of the evaluation checks that passed.
    """
    if not results:
        return 0.0

    return sum(results) / len(results)


def evaluate_retriever(
    eval_questions: list[EvalQuestion], retrieve_fn: Callable[[str], list[RetrievalResult]]
) -> list[bool]:
    """
    Evaluate a retreiver against the expected terms. And we can pass in any
    retriever function (i.e. retrieve_chunk, retrieve_semantic_chunk, etc) and
    reuse the same method instead of creating it again and again everywhere.
    """
    results = []
    for item in eval_questions:
        question = item["question"]
        expected_terms = item["expected_terms"]
        if not isinstance(question, str):
            raise TypeError("eval questions must be a str")
        if not isinstance(expected_terms, list):
            raise TypeError("expected_terms must be a list")

        retrieved_chunks = retrieve_fn(question)
        success = contains_expected_terms(
            retrieved_chunks=retrieved_chunks, expected_terms=expected_terms
        )
        results.append(success)
    return results


def format_accuracy_summary(
    method_name: str,
    results: list[bool],
) -> dict[str, str]:
    """
    Format evaluation results for summary reporting.
    """
    passed = str(sum(results))
    total = str(len(results))
    accuracy = calculate_accuracy(results=results)

    return {"method": method_name, "passed": passed, "total": total, "accuracy": f"{accuracy:.2f}"}


def print_summary_table(summaries: list[dict[str, str]]) -> None:
    """
    Print a small table comparing retrieval methods.
    """
    print("\nRetrieval Method Comparison")
    print("---------------------------")
    print(f"{'Method':<20} {'Passed':<8} {'Total':<8} {'Accuracy':<8}")

    for summary in summaries:
        print(
            f"{summary['method']:<20} "
            f"{summary['passed']:<8} "
            f"{summary['total']:<8} "
            f"{summary['accuracy']:<8}"
        )
