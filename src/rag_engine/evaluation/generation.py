import json

from rag_engine.generator import OllamaGenerator
from rag_engine.models import FaithfulnessResult, RetrievedChunk

def build_faithfulness_prompt(question: str, answer: str, sources: list[RetrievedChunk]) -> str:
    context = "\n\n".join(result.chunk.text for result in sources)
    return f"""
You are evaluating the faithfulness of an answer produced by a retrieval-augmented general_after_validator_function
system.
Determine whether the clais in the answer are supported by the provided context.

Question:
    {question}
Context:
    {context}
Answer:
    {answer}

Score the answer from 0.0 to 1.0:
    1.0 = every factual claim is supported by the context
    0.5 = some claims are supported and some are unsupported
    0.0 = the answer is unsupported or contradicts the context

Do not evaluate whether the answer is generally/factually correct.
Evaluate only whether the answer is supported by the provided context.

Return ONLY valid JSON in this format:
    {{
        "score": 0.0,
        "reasoning": "short explanation"
    }}
""".strip()

def evaluate_faithfulness(question: str,
                          answer: str,
                          sources: list[RetrievedChunk],
                          evaluator: OllamaGenerator,
                          abstained: bool = False) -> FaithfulnessResult:
    if abstained:
        return FaithfulnessResult(score=1.0, reasoning="The system abstained from answering")
    prompt = build_faithfulness_prompt(question=question,
                                        answer=answer,
                                        sources=sources)
    response = evaluator.generate(prompt, json_format=True)
    data = json.loads(response)
    return FaithfulnessResult.model_validate(data)
