import pytest
from pydantic import ValidationError
from rag_engine.models import Chunk

def test_chunk_rejects_missing_required_fields() -> None:
    with pytest.raises(ValidationError):
        Chunk(
            id="doc-1::chunk-0",
            document_id="doc-1",
            text="Example",
            source="data/raw/example.txt",
            start_char=0,
        )
