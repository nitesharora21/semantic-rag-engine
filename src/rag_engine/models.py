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
