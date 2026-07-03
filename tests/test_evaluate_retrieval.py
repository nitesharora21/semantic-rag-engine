from scripts.evaluate_retrieval import contains_expected_terms
from rag_engine.evaluation import calculate_accuracy


def test_contains_expected_terms() -> None:
    retrieved_chunks = [
        (2, "Nitesh built Kafka-based event-driven services."),
    ]
    result = contains_expected_terms(
        retrieved_chunks=retrieved_chunks, expected_terms=["Kafka", "event-driven"]
    )

    assert result is True


def test_contains_expected_terms_returns_true_when_all_terms_exist() -> None:
    retrieved_chunks = [(2, "Nitesh built Kafka-based event-driven services.")]

    result = contains_expected_terms(
        retrieved_chunks=retrieved_chunks, expected_terms=["Kafka", "event-driven"]
    )
    assert result is True


def test_contains_expected_terms_returns_true_when_terms_are_missing() -> None:
    retrieved_chunks = [(1, "Nitesh built developer tooling")]
    result = contains_expected_terms(retrieved_chunks=retrieved_chunks, expected_terms=["Kafka"])
    assert result is False


def test_calculate_accuracy_returns_zero_for_empty_results() -> None:
    accuracy = calculate_accuracy([])
    assert accuracy == 0.0


def test_calculate_accuracy_returns_fraction_of_passed_results() -> None:
    results = [True, True, True, False]
    accuracy = calculate_accuracy(results)
    assert accuracy == 0.75
