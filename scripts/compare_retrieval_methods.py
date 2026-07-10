from rag_engine.evaluation import (
    evaluate_retriever,
    load_eval_questions,
    format_accuracy_summary,
    print_summary_table,
)
from evaluate_retrieval import keyword_retrieve
from evaluate_semantic_retrieval import semantic_retrieve
from evaluate_faiss_retrieval import faiss_retrieve


def main() -> None:
    eval_questions = load_eval_questions("eval/retrieval_questions.json")

    # List all the retriever methods here, add any new retriever_method here.
    retriever_methods_dict = {
        "Keyword": keyword_retrieve,
        "Manual Semantic": semantic_retrieve,
        "FAISS Semantic": faiss_retrieve,
    }

    # Generic function - no need to add anything here below
    retriever_results_dict = {}
    for retriever_method_name, retriever_method in retriever_methods_dict.items():
        retriever_results_dict[retriever_method_name] = evaluate_retriever(
            eval_questions=eval_questions,
            retrieve_fn=retriever_method,
        )

    summaries = []

    for retriever_method_name, retriever_results in retriever_results_dict.items():
        summaries.append(format_accuracy_summary(retriever_method_name, retriever_results))

    print_summary_table(summaries)


if __name__ == "__main__":
    main()
