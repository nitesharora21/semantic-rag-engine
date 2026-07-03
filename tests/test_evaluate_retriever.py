from scripts.evaluate_retrieval import contains_expected_terms


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
