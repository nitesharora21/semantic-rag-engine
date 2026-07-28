from rag_engine.chunking import chunk_document
from rag_engine.models import Document


def test_chunk_document_returns_one_chunk_for_short_text() -> None:
    text = "This is one short text testing for the chunk size"
    chunks = chunk_document(
        Document(
            id="doc-1",
            text=text,
            source="data/raw/example.txt",
        ),
        chunk_size=100,
    )
    assert chunks[0].model_dump()  == {
            "id": "doc-1:: chunk-0",
            "document_id": "doc-1",
            "text": "This is one short text testing for the chunk size",
            "source": "data/raw/example.txt",
            "start_char": 0,
            "end_char": 49
        }

def test_chunk_offsets_reconstruct_original_text() -> None:
    text = "abcdefghij"
    chunks = chunk_document(
        Document(
            id="doc-1",
            text=text,
            source="data/raw/example.txt",
        ),
        chunk_size=3,
    )
    for chunk in chunks:
        assert chunk.model_dump()['text'] == text[chunk.model_dump()['start_char']: chunk.model_dump()['end_char']]
