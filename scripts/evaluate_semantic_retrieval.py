import json

from pathlib import Path

from rag_engine.embeddings import EmbeddingModel
from rag_engine.evaluation import calculate_accuracy, contains_expected_terms
from rag_engine.semantic_retriever import retrieve_semantic_chunks
from rag_engine.storage import load_chunks, load_embeddings

def load_eval_questions(file_path: str) -> list[dict]:
  path = Path(file_path)
  return json.loads(path.read_text(encoding="utf-8"))

def main() -> None:
  chunks = load_chunks("data/processed/chunks.json")
  chunk_embeddings = load_embeddings("data/processed/embeddings.json")
  eval_questions = load_eval_questions("eval/retrieval_questions.json")

  model = EmbeddingModel()

  results_summary = []

  for item in eval_questions:
    question = item["question"]
    expected_terms = item["expected_terms"]

    query_embedding = model.embed_text(question)

    retrieved_chunks = retrieve_semantic_chunks(
      query_embedding=query_embedding,
      chunks=chunks,
      chunk_embeddings=chunk_embeddings,
      top_k=3
    )

    success = contains_expected_terms(retrieved_chunks, expected_terms)
    results_summary.append(success)

    status = "PASS" if success else "FAIL"

    print(f"\n [{status}] {question}")
    print(f"Expected terms: {expected_terms}")

    for index, result in enumerate(retrieved_chunks, start=1):
      score, chunk = result
      print(f"--- Result {index} | Similarity: {score:.4f}")
      print(f"\t{chunk[:300]}")

  passed = sum(results_summary)
  total = len(results_summary)
  accuracy = calculate_accuracy(results_summary)

  print(f"--- Final Summary ---")
  print(f"Passed: {passed}")
  print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
