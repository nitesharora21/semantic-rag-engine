from unittest.mock import Mock

from rag_engine.rag import RAGEngine


from rag_engine.models import (
    Chunk,
    RetrievedChunk,
)


def test_rag_abstains_when_retrieval_score_is_too_low() -> None:
    engine = RAGEngine.__new__(RAGEngine)

    chunk = Chunk(
        id="doc::chunk-0",
        document_id="doc",
        text="Some unrelated content.",
        source="example.txt",
        start_char=0,
        end_char=23,
    )

    engine.minimum_score = 0.35

    engine.retrieve = Mock(
        return_value=[
            RetrievedChunk(
                chunk=chunk,
                score=0.12,
            )
        ]
    )

    engine.generator = Mock()

    response = engine.answer(
        "What is the capital of Australia?"
    )

    assert response.abstained is True
    assert (
        response.answer
        == "I do not have enough information in the provided context."
    )

    engine.generator.generate.assert_not_called()
