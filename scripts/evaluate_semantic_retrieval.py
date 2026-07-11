from rag_engine.embeddings import EmbeddingModel
from rag_engine.semantic_retriever import retrieve_semantic_chunks
from rag_engine.storage import load_chunks, load_embeddings
from rag_engine.evaluation import (
    load_eval_questions,
    RetrievalResult,
)


chunks = load_chunks("data/processed/chunks.json")
chunk_embeddings = load_embeddings("data/processed/embeddings.json")
eval_questions = load_eval_questions("eval/retrieval_questions.json")
chunk_texts = [chunk['text'] for chunk in chunks]
chunk_id_by_text = {chunk['text']: chunk['id'] for chunk in chunks}

model = EmbeddingModel()


def semantic_retrieve(question: str) -> list[RetrievalResult]:
    query_embedding = model.embed_text(question)
    results = retrieve_semantic_chunks(
        query_embedding=query_embedding,
        chunks=chunks_texts,
        chunk_embeddings=chunk_embeddings,
        top_k=3,
    )
    return [(score, chunk_id_by_text[text]) for score, text in results]


def main() -> None:

    recall_scores = evaluate_recall_at_k(eval_questions=eval_questions, retrieve_fn=semantic_retrieve, k=3)
    for item, recall in zip(eval_questions, recall_scores):
        print(f"\nQuestion: {item['question']}")
        print(f"Expected chunks: {item['expected_chunk_ids']}")
        print(f"Recall@3: {recall:.2f}")
    mean_recall = calculate_mean_score(recall_scores)
    print("\n--- Summary ---")
    print(f"Mean Recall@3: {mean_recall:.2f}")

if __name__ == "__main__":
    main()
