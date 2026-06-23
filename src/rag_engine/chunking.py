def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """
    Split text into fixed-sized character size chunks, based on chunk_size.

    """
    chunks = []
    for start in range(0, len(text), chunk_size):
        end = start + chunk_size
        chunks.append(text[start:end])
    return chunks
