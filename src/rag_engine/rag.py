from rag_engine.context import select_context
from rag_engine.embeddings import EmbeddingModel
from rag_engine.generator import OllamaGenerator
from rag_engine.models import RetrievedChunk, RAGResponse
from rag_engine.prompting import build_rag_prompt
from rag_engine.storage import load_chunks
from rag_engine.vector_store import FaissVectorStore

class RAGEngine:
    def __init__(self, 
                 chunks_path: str, 
                 index_path: str, 
                 embedding_model: EmbeddingModel, 
                 generator: OllamaGenerator, 
                 top_k: int = 5,
                 minimum_score: float = 0.5,
                 max_context_chars: int = 4000) -> None:
        self.chunks = load_chunks(chunks_path)
        self.vector_store = FaissVectorStore.load(index_path)
        self.embedding_model = embedding_model
        self.generator = generator
        # Retrieval budget != Generation Context Budget (top_k does not impact context chars)
        self.top_k = top_k
        self.minimum_score = minimum_score
        self.max_context_chars = max_context_chars

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        query_embedding = self.embedding_model.embed_text(question)
        results = self.vector_store.search(query_embedding=query_embedding, top_k=self.top_k)
        return [RetrievedChunk(chunk=self.chunks[chunk_index], score=score) for score, chunk_index in results]
    
    def has_sufficient_score(self, results: list[RetrievedChunk]) -> bool:
        if not results:
            return False
        return results[0].score >= self.minimum_score
    
    def answer(self, question: str) -> RAGResponse:
        results = self.retrieve(question)
        if not self.has_sufficient_score(results):
            return RAGResponse(
                    question=question,
                    answer=("I do not have enough information in the provided context."),
                    abstained=True,
                    sources=results)
        context_results = select_context(results=results, max_chars=self.max_context_chars)
        chunks = [result.chunk for result in context_results]
        prompt = build_rag_prompt(question=question, chunks=chunks)
        answer = self.generator.generate(prompt)
        return RAGResponse(
                question=question,
                answer=answer,
                abstained=False,
                sources=context_results)
    


