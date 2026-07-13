import json
from pathlib import Path
from typing import Callable, TypedDict

RetrievalResult = tuple[int | float, str]  # Generally score and chunk


class EvalQuestion(TypedDict):
    question: str
    expected_terms: list[str]


def calculate_reciprocal_rank(retrieved_chunk_ids: list[str], expected_chunk_ids: list[str]) -> float:
    """
    Each eval question will have the expected_terms (i.e. expected_chunk_ids) and the
    retrieved_chunk_ids.
    So we calculate the RR using the retrieved_chunks.
    NOTE: The retrieved chunks are already sorted by their scores, so the highest score chunk is
    the first one in the list of retrieved_chunks.
    NOTE: The score is based on the position of the first relevant chunk only.
    """
    expected_chunk_ids = list(set(expected_chunk_ids))
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in expected_chunk_ids:
            return 1 / rank
    return 0.0

def evaluate_reciprocal_rank(eval_questions: list[EvalQuestion], retrieve_fn: Callable[[str], list[RetrievalResult]]) -> list[float]:
    scores = []
    for item in eval_questions:
        retrieved_chunks = retrieve_fn(str(item['question']))
        retrieved_chunk_ids = [chunk_id for _, chunk_id in retrieved_chunks]
        score = calculate_reciprocal_rank(retrieved_chunk_ids=retrieved_chunk_ids, expected_chunk_ids=item['expected_terms'])
        scores.append(score)
    return scores

def calculate_recall_at_k(retrieved_chunk_ids: list[str], expected_terms: list[str], k: int) -> float:
    """
    Calculating recall@k for one evaluation question.

    Definition: Recall@k is the fraction of the expected chunks over
    first k received results
    """
    if not expected_terms:
        return 0

    top_k_ids = retrieved_chunk_ids[:k]
    relevant_retrieved = set(top_k_ids) & set(expected_terms)
    return len(relevant_retrieved) / len(expected_terms)


def evaluate_recall_at_k(eval_questions: list[EvalQuestion], retrieve_fn: Callable[[str], list[RetrievalResult]], k: int = 3) -> list[float]:
    """45
    Calculate recall@k for every eval question now
    For each question retrieve the top k chunks for the type of retrieval_method provided
    Then with the retrieved_chunks data - extract the chunk ids from it
    And then use the retrieved_chunk_ids and expected_terms from the item
    to compute the recall@k
    """
    scores = []
    for item in eval_questions:
        retrieved_chunks = retrieve_fn(item["question"])
        retrieved_chunk_ids = [chunk_id for _, chunk_id in retrieved_chunks]
        score = calculate_recall_at_k(
            retrieved_chunk_ids=retrieved_chunk_ids, expected_terms=item["expected_terms"], k=k
        )
        scores.append(score)
    return scores


def calculate_mean_score(scores: list[float]) -> float:
    if not scores: return 0.0
    return sum(scores) / len(scores)


def contains_expected_terms(retrieved_chunks: list[RetrievalResult], expected_terms: list[str]) -> bool:
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


def evaluate_retriever(eval_questions: list[EvalQuestion], retrieve_fn: Callable[[str], list[RetrievalResult]]) -> list[bool]:
    """
    Evaluate a retreiver against the expected terms. And we can pass in any
    retriever function (i.e. retrieve_chunk, retrieve_semantic_chunk, etc) and
    reuse the same method instead of creating it again and again everywhere.
    """
    results = []
    for item in eval_questions:
        question = item["question"]
        expected_terms = item["expected_terms"]
        if not isinstance(question, str): raise TypeError("eval questions must be a str")
        if not isinstance(expected_terms, list): raise TypeError("expected_terms must be a list")
        retrieved_chunks = retrieve_fn(question)
        success = contains_expected_terms(retrieved_chunks=retrieved_chunks, expected_terms=expected_terms)
        results.append(success)
    return results


def load_eval_questions(file_path: str) -> list[EvalQuestion]:
    """
    Load the evaluation questions with their expected terms and return as a json dict"""
    path = Path(file_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def calculate_accuracy(results: list[bool]) -> float:
    """
    Calculate the fraction of the evaluation checks that passed.
    """
    if not results: return 0.0
    return sum(results) / len(results)


def format_accuracy_summary(method_name: str, results: list[bool]) -> dict[str, str]:
    """
    Format evaluation results for summary reporting.
    """
    passed = str(sum(results))
    total = str(len(results))
    accuracy = calculate_accuracy(results=results)
    return {"method": method_name, "passed": passed, "total": total, "accuracy": f"{accuracy:.2f}"}

def format_retrieval_metrics(method_name: str, recall_scores: list[float], reciprocal_rank_scores: list[float]) -> dict[str, str]:
    return {
        "method": method_name,
        "mean_recall": f"{calculate_mean_score(recall_scores):.2f}",
        "mrr": f"{calculate_mean_score(reciprocal_rank_scores):.2f}"
    }

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


def print_retrieval_metrics_table(summaries: list[dict[str, str]], k: int = 3) -> None:
    """
    Print recall@k and MRR for multiple retrieval methods.
    """
    print("\nRetrieval Method Comparison")
    print("---------------------------")
    print(
        f"{'Method':<20} "
        f"{f'Mean Recall@{k}':<16} "
        f"{'MRR':<8}"
    )

    for summary in summaries:
        print(
            f"{summary['method']:<20} "
            f"{summary['mean_recall']:<16} "
            f"{summary['mrr']:<8}"
        )
