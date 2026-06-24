from rag_engine.chunking import chunk_text
from rag_engine.loaders import load_text_file

CHUNK_SIZE = 300

def main() -> None:
    text = load_text_file(f"../data/raw/resume_profile.txt")
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE)

    for index, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {index} ---")
        print(chunk)

if __name__ == "__main__":
    main()
