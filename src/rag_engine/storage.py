import json
from pathlib import Path

def save_chunks(chunks: list[str], output_path: str) -> None:
    """
    Take the chunks and store it to the JSON file in the output path.

    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
