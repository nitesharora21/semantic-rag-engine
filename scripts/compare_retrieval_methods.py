import json
from pathlib import Path

from rag_engine.embeddings import EmbeddingModel
from rag_engine.vector_store import FaissVectorStore
from rag_engine.storage import load_chunks, load_embeddings
from rag_engine.retriever import retrieve_chunks
from rag_engine.semantic_retriever import retrieve_semantic_chunks
from rag_engine.evaluation import contains_expected_terms, format_accuracy_summary


def load_eval_questions(input_path: str) -> list[dict]:
    path = Path(input_path)
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_keyword(eval_questions: list[dict], chunks: list[str]) -> list[bool]:
    results_summary = []
    for item in eval_questions:
        question = item["question"]
        expected_terms = item["expected_terms"]
        retrieved_chunks = retrieve_chunks(query=question, chunks=chunks, top_k=3)
        success = contains_expected_terms(
            retrieved_chunks=retrieved_chunks, expected_terms=expected_terms
        )
        results_summary.append(success)
    return results_summary


def evaluate_manual_semantic(
    eval_questions: list[dict],
    chunks: list[str],
    embeddings: list[list[float]],
    model: EmbeddingModel,
) -> list[bool]:
    results = []
    for item in eval_questions:
        question = item["question"]
        expected_terms = item["expected_terms"]

        retrieved_chunks = retrieve_semantic_chunks(
            query_embedding=model.embed_text(question),
            chunks=chunks,
            chunk_embeddings=embeddings,
            top_k=3,
        )

        success = contains_expected_terms(
            retrieved_chunks=retrieved_chunks,
            expected_terms=expected_terms,
        )
        results.append(success)
    return results


def evaluate_faiss_results(
    eval_questions: list[dict], chunks: list[str], model: EmbeddingModel, store: FaissVectorStore
) -> list[bool]:
    results_summary = []
    for item in eval_questions:
        question = item["question"]
        expected_terms = item["expected_terms"]
        query_embedding = model.embed_text(question)
        results = store.search(query_embedding=query_embedding, top_k=3)
        retrieved_chunks = [(score, chunks[chunk_index]) for score, chunk_index in results]
        success = contains_expected_terms(
            retrieved_chunks=retrieved_chunks, expected_terms=expected_terms
        )
        results_summary.append(success)
    return results_summary


def print_summary_table(summaries: list[dict[str, str]]) -> None:
    print("\n--- Retrieval Method Comparison ---")
    print("-----------------------------------")
    print(f"{'Method': <20} {'Passed': <8} {'Total': <8} {'Accuracy': <8}")

    for summary in summaries:
        print(
            f"{summary['method']: <20} "
            f"{summary['passed']: <8} "
            f"{summary['total']: <8} "
            f"{summary['accuracy']: <8} "
        )


def main() -> None:
    chunks = load_chunks("data/processed/chunks.json")
    embeddings = load_embeddings("data/processed/embeddings.json")
    eval_questions = load_eval_questions("eval/retrieval_questions.json")

    model = EmbeddingModel()
    store = FaissVectorStore(embeddings=embeddings)

    keyword_results = evaluate_keyword(
        eval_questions=eval_questions,
        chunks=chunks,
    )

    manual_semantic_results = evaluate_manual_semantic(
        eval_questions=eval_questions,
        chunks=chunks,
        embeddings=embeddings,
        model=model,
    )

    faiss_results = evaluate_faiss_results(
        eval_questions=eval_questions,
        chunks=chunks,
        model=model,
        store=store,
    )

    summaries = [
        format_accuracy_summary("Keyword", keyword_results),
        format_accuracy_summary("Manual Semantic", manual_semantic_results),
        format_accuracy_summary("FAISS Semantic", faiss_results),
    ]

    print_summary_table(summaries)


if __name__ == "__main__":
    main()
