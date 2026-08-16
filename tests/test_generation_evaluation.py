from unittest.mock import Mock

from rag_engine.evaluation.generation import evaluate_faithfulness
from rag_engine.models import Chunk, RetrievedChunk


def test_evaluate_faithfulness_parses_score() -> None:
    evaluator = Mock()

    evaluator.generate.return_value = """ 
{ 
	"score": 1.0, 
	"reasoning": "The answer is supported."
}
    """

    source = RetrievedChunk(chunk=Chunk(id="doc::chunk-0",
										document_id="doc", 
										text="Kafka is used for event streaming.", 
										source="example.txt",
										start_char=0,
										end_char=34),
							score=0.9)

    result = evaluate_faithfulness(question="What is Kafka used for?",
								   answer="Kafka is used for event streaming.",
								   sources=[source],
								   evaluator=evaluator)

    assert result.score == 1.0
