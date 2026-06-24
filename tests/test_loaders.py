from rag_engine.loaders import load_text_file


def test_load_text_file_reads_file_content(tmp_path) -> None:
    file_path = tmp_path / "example.txt"
    file_path.write_text(
        "Hello from the other side - I must have called a thousand times.", encoding="utf-8"
    )

    content = load_text_file(str(file_path))

    assert content == "Hello from the other side - I must have called a thousand times."
