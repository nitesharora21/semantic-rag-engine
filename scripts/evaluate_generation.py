import json

from rag_engine.embeddings import EmbeddingModel
from rag_engine.evaluation.generation import evaluate_faithfulness
from rag_engine.generator import OllamaGenerator
from rag_engine.rag import RAGEngine

def main() -> None:
    with open("eval/retrieval_questions.json", encoding="utf-8") as file:
        questions = json.load(file)
    generator = OllamaGenerator(model_name="gemma3")
    rag = RAGEngine(chunks_path="data/processed/chunks.json",
                    index_path="data/processed/faiss.index",
                    embedding_model=EmbeddingModel(),
                    generator=generator,
                    top_k=5,
                    minimum_score=0.35,
                    max_context_chars=4000)
    faithfulness_scores : list[float] = []
    for item in questions:
        question = item['question']
        response = rag.answer(question)
        evaluation = evaluate_faithfulness(question=question,
                                            answer=response.answer,
                                            sources=response.sources,
                                            evaluator=generator,
                                            abstained=response.abstained)
        faithfulness_scores.append(evaluation.score)
        print(f"\nQuestion: {question}")
        print(f"\nAnswer: {response.answer}")
        print(f"\nFaithfulness: {evaluation.score:.2f}")
        print(f"\nReasoning: {evaluation.reasoning}")
        
        mean_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)

        print("\n=================")

        print(f"Mean Faithfulness: {mean_faithfulness:.3f}")

if __name__ == "__main__":
    main()

