import json
from pathlib import Path

from rag_engine.embeddings import EmbeddingModel
from rag_engine.storage import load_chunks, load_embeddings
from rag_engine.vector_store import FaissVectorStore
from rag_engine.evaluation import calculate_accuracy, contains_expected_terms

def load_eval_questions(file_path: str) -> list[dict]:
  path = Path(file_path)
  return json.loads(path.read_text(encoding="utf-8"))

def main()-> None:
  chunks = load_chunks("data/processed/chunks.json")
  embeddings = load_embeddings("data/processed/embeddings.json")
  eval_questions = load_eval_questions("eval/retrieval_questions.json")

  model = EmbeddingModel()
  store = FaissVectorStore(embeddings=embeddings)

  results_summary = []

  for item in eval_questions:
    question = item["question"]
    expected_terms = item['expected_terms']

    query_embedding = model.embed_text(question)
    faiss_results = store.search(query_embedding=query_embedding, top_k=3)

    retrieved_chunks = [
      (score, chunks[chunk_index]) for score, chunk_index in faiss_results
    ]

    success = contains_expected_terms(retrieved_chunks=retrieved_chunks,
                                      expected_terms=expected_terms)
    results_summary.append(success)

    status = "PASS" if success else "FAIL"
    print(f"[{status}] {question}")
    print(f"Expected Terms: {expected_terms}")

    for index, result in enumerate(retrieved_chunks, start=1):
      score, chunk = result
      print(f"---\n Result {index} | FAISS Score: {score:.4f}")
      print(f"\t{chunk[:300]}")

  passed = sum(results_summary)
  total = len(results_summary)
  accuracy = calculate_accuracy(results_summary)

  print(f"\n--- Summary ---")
  print(f"Passed: {passed}")
  print(f"Total: {total}")
  print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
  main()
