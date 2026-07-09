from collections.abc import Sequence


def contains_expected_terms(
    retrieved_chunks: Sequence[tuple[int | float, str]], expected_terms: list[str]
) -> bool:
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


def calculate_accuracy(results: list[bool]) -> float:
    """
    Calculate the fraction of the evaluation checks that passed.
    """
    if not results:
        return 0.0

    return sum(results) / len(results)


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
