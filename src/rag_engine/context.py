from rag_engine.models import RetrievedChunk

def select_context(results: list[RetrievedChunk], max_chars: int) -> list[RetrievedChunk]:
    """
    Deduplicate retrieved chunks. Keep ranked context with char limit
    Check on actual text, since seperate chunk id can contain same texts.
    """
    selected: list[RetrievedChunk] = []
    seen_texts: set[str] = set()
    total_chars = 0
    for result in results:
        normalized_text = result.chunk.text.strip()
        if not normalized_text:
            continue
        if normalized_text in seen_texts:
            continue
        chunk_size = len(normalized_text)
        if total_chars + chunk_size > max_chars:
            continue
        selected.append(result)
        seen_texts.add(normalized_text)
        total_chars += chunk_size
    return selected
