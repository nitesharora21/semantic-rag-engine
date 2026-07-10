from rag_engine.embeddings import EmbeddingModel
from rag_engine.semantic_retriever import retrieve_semantic_chunks
from rag_engine.storage import load_chunks, load_embeddings
from rag_engine.evaluation import (
    calculate_accuracy,
    evaluate_retriever,
    load_eval_questions,
    RetrievalResult,
)


chunks = load_chunks("data/processed/chunks.json")
chunk_embeddings = load_embeddings("data/processed/embeddings.json")
eval_questions = load_eval_questions("eval/retrieval_questions.json")

model = EmbeddingModel()


def semantic_retrieve(question: str) -> list[RetrievalResult]:
    query_embedding = model.embed_text(question)
    return retrieve_semantic_chunks(
        query_embedding=query_embedding,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        top_k=3,
    )


def main() -> None:

    results_summary = evaluate_retriever(
        eval_questions=eval_questions,
        retrieve_fn=semantic_retrieve,
    )

    for item, success in zip(eval_questions, results_summary):
        question = item["question"]
        expected_terms = item["expected_terms"]
        retrieved_chunks = semantic_retrieve(str(question))
        status = "SUCCESS" if success else "FAIL"
        print(f"\n [{status}] {question}")
        print(f"Expected Terms: {expected_terms}")
        for index, result in enumerate(retrieved_chunks, start=1):
            score, chunk = result
            print(f"\n Result {index} | Similarity: {score:.4f}")
            print(f"\t{chunk[:300]}")

    passed = sum(results_summary)
    total = len(results_summary)
    accuracy = calculate_accuracy(results_summary)

    print("\n--- Summary ---")
    print(f"Passed: {passed / total}")
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
