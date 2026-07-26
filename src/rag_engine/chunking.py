from typing import TypedDict
from rag_engine.models import Document, Chunk

def chunk_document(document: Document, chunk_size: int = 500) -> list[Chunk]:
    """
    Split text into fixed-sized character size chunks, based on chunk_size.
    Updates:
    1. 2026-07-10: Modified chunks from a simple list to a list of dict with index and text
    2. 2026-07-16: Added meta-data to the chunks, id, text, document_id, source etc.
    3.2026-07-26: Modified chunk_text -> chunk_document, using pydantic models and used Document attributes instead of params
    """
    chunks: list[Chunk] = []

    for index, start in enumerate(range(0, len(document.text), chunk_size)):
        # To ensure end char does not exceed len(text), actual document length
        end = min(start + chunk_size, len(document.text))
        chunks.append(
            Chunk(
                id=f"{document.id}:: chunk-{index}",
                document_id=document.id,
                text=str(document.text[start:end]),
                source=document.source,
                start_char=start,
                end_char=end,
            )
        )
    return chunks
