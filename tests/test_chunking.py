from rag_engine.chunking import chunk_text


def test_chunk_text_returns_one_chunk_for_short_text() -> None:
    text = "This is one short text testing for the chunk size"

    chunks = chunk_text(text, chunk_size=100)

    assert chunks, "This is one short text testing for the chunk size"
