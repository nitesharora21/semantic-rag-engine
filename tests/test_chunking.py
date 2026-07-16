from rag_engine.chunking import chunk_text


def test_chunk_text_returns_one_chunk_for_short_text() -> None:
    text = "This is one short text testing for the chunk size"
    chunks = chunk_text(
        text=text,
        document_id="doc-1",
        source="data/raw/example.txt",
        chunk_size=100,
    )
    assert chunks == [
        {
            "id": "doc-1:: chunk-0",
            "document_id": "doc-1",
            "text": "This is one short text testing for the chunk size",
            "source": "data/raw/example.txt",
            "start_char": 0,
            "end_char": 49
        }
    ]

def test_chunk_offsets_reconstruct_original_text() -> None:
    text = "abcdefghij"
    chunks = chunk_text(
        text=text,
        document_id="doc-1",
        source="data/raw/example.txt",
        chunk_size=3,
    )
    for chunk in chunks:
        assert chunk['text'] == text[chunk['start_char']: chunk['end_char']]
