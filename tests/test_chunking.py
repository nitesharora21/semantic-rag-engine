from rag_engine.chunking import chunk_text


def test_chunk_text_returns_one_chunk_for_short_text() -> None:
    text = "This is one short text testing for the chunk size"
    chunks = chunk_text(text, chunk_size=100)
    assert chunks == [
        {"id": "chunk-0", "text": "This is one short text testing for the chunk size"}
    ]


def test_chunk_text_splits_text_into_multiple_chunks() -> None:
    text = "abcdefghijk"
    chunks = chunk_text(text=text, chunk_size=3)
    assert chunks == [
        {"id": "chunk-0", "text": "abc"},
        {"id": "chunk-1", "text": "def"},
        {"id": "chunk-2", "text": "ghi"},
        {"id": "chunk-3", "text": "jk"},
    ]
