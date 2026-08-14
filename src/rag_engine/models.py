from pydantic import BaseModel

class Document(BaseModel):
    id: str
    text: str
    source: str


class Chunk(BaseModel):
    id: str
    document_id: str
    text: str
    source: str
    start_char: int
    end_char: int

class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float

class RAGResponse(BaseModel):
    question: str
    answer: str
    abstained: bool
    sources: list[RetrievedChunk]
