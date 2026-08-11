from unittest.mock import Mock

from rag_engine.rag import RAGEngine


def test_rag_answer_uses_retrieved_context(monkeypatch,) -> None:
    engine = RAGEngine.__new__(RAGEngine)

    chunk = Mock()
    chunk.id = "doc::chunk-0"
    chunk.source = "example.txt"
    chunk.text = "Kafka is used for event-driven services."

    engine.retrieve = Mock(return_value=[chunk])

    engine.generator = Mock()
    engine.generator.generate.return_value = "Generated answer"

    answer = engine.answer("What is Kafka used for?")

    assert answer == "Generated answer"

    engine.generator.generate.assert_called_once()

    prompt = engine.generator.generate.call_args.args[0]

    assert "Kafka is used for event-driven services." in prompt
    assert "What is Kafka used for?" in prompt
