from rag_engine.embeddings import EmbeddingModel
from rag_engine.generator import OllamaGenerator
from rag_engine.models import Chunk
from rag_engine.prompting import build_rag_prompt
from rag_engine.storage import load_chunks
from rag_engine.vector_store import FaissVectorStore

class RAGEngine:
    def __init__(self, 
                 chunks_path: str, 
                 index_path: str, 
                 embedding_model: EmbeddingModel, 
                 generator: OllamaGenerator, 
                 top_k: int = 3) -> None:
        self.chunks = load_chunks(chunks_path)
        self.vector_store = FaissVectorStore.load(index_path)
        self.embedding_model = embedding_model
        self.generator = generator
        self.top_k = top_k

    def retrieve(self, question: str) -> list[Chunk]:
        query_embedding = self.embedding_model.embed_text(question)
        results = self.vector_store.search(query_embedding=query_embedding, top_k=self.top_k)
        return [self.chunks[chunk_index] for _, chunk_index in results]
    
    def answer(self, question: str) -> str:
        chunks = self.retrieve(question)
        prompt = build_rag_prompt(question=question, chunks=chunks)
        return self.generator.generate(prompt)
    


