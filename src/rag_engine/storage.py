import json
from pathlib import Path
from rag_engine.models import Chunk


def save_chunks(chunks: list[Chunk], output_path: str) -> None:
    """
    Take the chunks and store it to the JSON file in the output path.

    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert the list of chunk data objects into a list of dict using model_dump()
    chunk_data_json = [chunk.model_dump() for chunk in chunks]
    path.write_text(json.dumps(chunk_data_json, indent=2), encoding="utf-8")


def load_chunks(input_path: str) -> list[Chunk]:
    """
    Load texts from JSON file and converts them into Chunk BaseModel
    """
    path = Path(input_path)
    content = path.read_text(encoding="utf-8")
    chunk_data_json = json.loads(content)
    return [Chunk.model_validate(chunk_json) for chunk_json in chunk_data_json]


def save_embeddings(embeddings: list[list[float]], output_path: str) -> None:
    """
    Save the embeddings to a JSON file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(embeddings), encoding="utf-8")


def load_embeddings(input_path: str) -> list[list[float]]:
    """
    Load the embeddings using the input_path provided.
    """
    path = Path(input_path)
    content = path.read_text(encoding="utf-8")
    return json.loads(content)
