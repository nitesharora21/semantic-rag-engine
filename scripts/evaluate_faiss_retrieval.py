from rag_engine.embeddings import EmbeddingModel
from rag_engine.storage import load_chunks, load_embeddings
from rag_engine.vector_store import FaissVectorStore
from rag_engine.evaluation import (
    calculate_accuracy,
    load_eval_questions,
    evaluate_retriever,
    RetrievalResult,
)

chunks = load_chunks("data/processed/chunks.json")
embeddings = load_embeddings("data/processed/embeddings.json")
eval_questions = load_eval_questions("eval/retrieval_questions.json")

model = EmbeddingModel()
store = FaissVectorStore(embeddings=embeddings)


def faiss_retrieve(question: str) -> list[RetrievalResult]:
    query_embedding = model.embed_text(question)
    retrieved_chunks = store.search(query_embedding=query_embedding, top_k=3)
    results = []
    for score, chunk_index in retrieved_chunks:
        for chunk_data in chunks:
            if chunk_data["id"] == f"chunk-{chunk_index}":
                results.append((score, chunk_data["text"]))
                break
    return results


def main() -> None:

    results_summary = evaluate_retriever(eval_questions=eval_questions, retrieve_fn=faiss_retrieve)

    for item, success in zip(eval_questions, results_summary):
        question = item["question"]
        expected_terms = item["expected_terms"]
        retreived_chunks = faiss_retrieve(question=str(question))

        status = "PASSED" if success else "FAIL"
        print(f"\n[{status}] {question}")
        print(f"Expected Terms: {expected_terms}")

        for index, result in enumerate(retreived_chunks, start=1):
            score, chunk = result
            print(f"\n Result {index} | FAISS Score: {score}")
            print(f"\t {chunk[:300]}")

    passed = sum(results_summary)
    total = len(results_summary)
    accuracy = calculate_accuracy(results_summary)

    print("\n--- Summary ---")
    print(f"Passed: {passed}")
    print(f"Total: {total}")
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
