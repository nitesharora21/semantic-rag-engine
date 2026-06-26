from pathlib import Path


def load_text_file(file_path: str) -> str:
    """
    Load text from local file.
    """
    path = Path(file_path)
    return path.read_text(encoding="utf-8")
