import json

from rag_engine.storage import save_chunks, load_chunks


def test_save_chunks_write_to_json_file(tmp_path) -> None:
    output_path = tmp_path / "chunks.json"
    chunks = ["first chunk", "second chunk"]
    save_chunks(chunks, str(output_path))

    saved_chunks = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved_chunks == ["first chunk", "second chunk"]


def test_load_chunks_from_json_file(tmp_path) -> None:
    input_path = tmp_path / "chunks.json"
    input_path.write_text(json.dumps(["first chunk", "second chunk"]), encoding="utf-8")
    chunks = load_chunks(str(input_path))
    assert chunks == ["first chunk", "second chunk"]
