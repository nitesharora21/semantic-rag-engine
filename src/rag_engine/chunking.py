def chunk_text(text: str, chunk_size: int = 500) -> list[dict[str, str]]:
    """
    Split text into fixed-sized character size chunks, based on chunk_size.
    Updates:
    1. 2026-07-10: Modified chunks from a simple list to a list of dict with index and text

    """
    chunks = []
    index = 0
    for start in range(0, len(text), chunk_size):
        end = start + chunk_size
        chunks.append({"id": f"chunk-{index}", "text": text[start:end]})
        index += 1
    return chunks
