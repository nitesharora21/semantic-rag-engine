import json

from rag_engine.evaluation import (
    calculate_accuracy,
    contains_expected_terms,
    load_eval_questions,
    format_accuracy_summary,
)


def test_load_eval_questions_reads_from_json_file(tmp_path) -> None:
    eval_path = tmp_path / "retrieval_questions.json"
    eval_path.write_text(
        json.dumps(
            [
                {
                    "question": "Tell me what does Nitesh know about Kafka?",
                    "expected_terms": ["Kafka"],
                }
            ]
        ),
        encoding="utf-8",
    )
    questions = load_eval_questions(str(eval_path))

    assert questions == [
        {
            "question": "Tell me what does Nitesh know about Kafka?",
            "expected_terms": ["Kafka"],
        }
    ]
    return


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


def test_format_accuracy_summary_returns_display_fields() -> None:
    summary = format_accuracy_summary(method_name="Keyword", results=[True, False, True])

    assert summary == {"method": "Keyword", "passed": "2", "total": "3", "accuracy": "0.67"}
