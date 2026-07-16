from typing import TypedDict

class Chunk(TypedDict):
    id: str
    document_id: str
    text: str
    source: str
    start_char: str
    end_char: str

def chunk_text(
    text: str,
    document_id: str,
    source: str,
    chunk_size: int = 500
    ) -> list[dict[str, str]]:
    """
    Split text into fixed-sized character size chunks, based on chunk_size.
    Updates:
    1. 2026-07-10: Modified chunks from a simple list to a list of dict with index and text
    2. 2026-07-16: Added meta-data to the chunks, id, text, document_id, source etc.

    """
    chunks: list[Chunk] = []

    for index, start in enumerate(range(0, len(text), chunk_size)):
        # To ensure end char does not exceed len(text), actual document length
        end = min(start + chunk_size, len(text))
        chunks.append(
            {
                "id": f"{document_id}:: chunk-{index}",
                "document_id": document_id,
                "text": str(text[start:end]),
                "source": source,
                "start_char": start,
                "end_char": end,
            }
        )
    return chunks
