import json

from rag_engine.evaluation import (
    calculate_accuracy,
    contains_expected_terms,
    load_eval_questions,
    format_accuracy_summary,
    calculate_mean_score,
    calculate_recall_at_k,
    evaluate_recall_at_k
)

def test_calculate_recall_at_k_returns_one_when_all_relevant_chunks_are_found() -> None:
    score = calculate_recall_at_k(
        retrieved_chunk_ids=["chunk-4", "chunk-2", "chunk-3"],
        expected_chunk_ids=["chunk-2", "chunk-3"],
        k=3
    )
    assert score == 1.0

def test_calculate_recall_at_k_returns_partial_score() -> None:
    score = calculate_recall_at_k(
        retrieved_chunk_ids=["chunk-4", "chunk-2", "chunk-8"]
        expected_chunk_ids=["chunk-2", "chunk-3"]
    )
    assert score == 0.5

def test_calculate_recall_at_k_respects_k() -> None:
    score = calculate_recall_at_k(
        retrieved_chunk_ids=["chunk-4", "chunk-2", "chunk-3"],
        expected_chunk_ids=["chunk-2", "chunk-3"]
        k=2
    )
    # We only look at the first k chunks in the retrieved_chunks_ids,
    # so here we ignore chunk-3 in retrieved_chunk_ids
    assert score == 0.5

def test_evaluate_recall_at_k_returns_score_for_each_question() -> None:
    eval_questions = [
        {
            "question": "What has Nitesh done in Kafka?"
            "expected_chunk_ids": ['chunk-5']
        },
        {
            "question": "What is Nitesh's recent education?"
            "expected_chunk_ids": ["chunk-10"]
        },
    ]
    def fake_retrieve(question: str) -> list[tuple[float, str]]:
        if "Kafka" in question:
            return [(0.9, "chunk-5")]
        else:
            return [(0.7, "chunk-10")]

    scores = evaluate_recall_at_k(
        eval_questions=eval_questions,
        retrieve_fn=fake_retrieve,
        k=3)
    assert scores == [1.0, 1.0]


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
