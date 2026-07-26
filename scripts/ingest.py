from rag_engine.chunking import chunk_document
from rag_engine.loaders import load_text_file
from rag_engine.models import Document
from rag_engine.storage import save_chunks

CHUNK_SIZE = 300
SOURCE_PATH="data/raw/resume_profile.txt"
DEST_PATH="data/processed/chunks.json"
DOCUMENT_ID="resume-profile"


def main() -> None:
    text = load_text_file(SOURCE_PATH)
    document_object = Document(
        id=DOCUMENT_ID,
        text=text,
        source=SOURCE_PATH,
    )
    chunks = chunk_document(
    document=document_object,
    chunk_size=CHUNK_SIZE)

    save_chunks(chunks, DEST_PATH)
    print(f"Saved {len(chunks)} to {DEST_PATH}")

if __name__ == "__main__":
    main()
