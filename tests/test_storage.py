import json

from rag_engine.storage import (
    save_chunks,
    load_chunks,
    save_embeddings,
    load_embeddings,
)


def test_save_chunks_write_to_json_file(tmp_path) -> None:
    output_path = tmp_path / "chunks.json"
    chunks = [{"chunk-0": "first chunk", "chunk-1": "second chunk"}]
    save_chunks(chunks, str(output_path))
    saved_chunks = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved_chunks == chunks


def test_load_chunks_from_json_file(tmp_path) -> None:
    input_path = tmp_path / "chunks.json"
    chunk_data = [{"chunk-0": "first chunk", "chunk-1": "second chunk"}]
    input_path.write_text(json.dumps(chunk_data), encoding="utf-8")
    chunks = load_chunks(str(input_path))
    assert chunks == chunk_data


def test_save_embeddings_writes_embeddings_to_json_file(tmp_path) -> None:
    output_path = tmp_path / "embeddings.json"
    embeddings = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    save_embeddings(embeddings, str(output_path))
    saved_embeddings = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved_embeddings == embeddings


def test_load_embeddings_reads_embeddings_from_json_file(tmp_path) -> None:
    input_path = tmp_path / "embeddings.json"
    embeddings = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    input_path.write_text(json.dumps(embeddings))
    loaded_embeddings = load_embeddings(str(input_path))
    assert loaded_embeddings == embeddings
