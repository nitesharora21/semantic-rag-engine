from rag_engine.context import select_context
from rag_engine.models import Chunk, RetrievedChunk

def make_result(chunk_id: str, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(id=chunk_id,
                                      document_id="doc",
                                      text=text,
                                      source="example.txt",
                                      start_char=0,
                                      end_char=len(text)),
                          score=score)

def test_select_context_deduplicates_and_respects_budget() -> None:
    results = [make_result("chunk-1", "Kafka event streaming", 0.90),
               make_result("chunk-2", "Kafka event streaming", 0.80),
               make_result("chunk-3", "FAISS vector search", 0.70)]
    selected = select_context(results, max_chars=40)
    assert [result.chunk.id for result in selected] == ["chunk-1", "chunk-3"]
