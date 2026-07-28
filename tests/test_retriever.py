from rag_engine.retriever import retrieve_chunks
from rag_engine.models import Chunk


def test_retreive_chunks_find_matching_chunks() -> None:
    chunks = [
        Chunk(id="doc-1:: chunk-0", document_id="doc-1", text="Nitesh has experience working with Kafka", source="data/raw/example.txt", start_char=0, end_char=39),
        Chunk(id="doc-1:: chunk-1", document_id="doc-1", text="He worked in build automation", source="data/raw/example.txt", start_char=40, end_char=69),
        Chunk(id="doc-1:: chunk-2", document_id="doc-1", text="He studied masters in computer science from Georgia Tech", source="data/raw/example.txt", start_char=70, end_char=126),
    ]
    results = retrieve_chunks("Kafka", chunks, top_k=1)
    assert results[0][1] == "Nitesh has experience working with Kafka"


def test_retrieve_chunks_top_k() -> None:
    query = "Nitesh"
    chunks = [
        Chunk(id="doc-1:: chunk-0", document_id="doc-1", text="Nitesh has experience working with Kafka.", source="data/raw/example.txt", start_char=0, end_char=39),
        Chunk(id="doc-1:: chunk-1", document_id="doc-1", text="He worked in build automation", source="data/raw/example.txt", start_char=40, end_char=69),
        Chunk(id="doc-1:: chunk-2", document_id="doc-1", text="He studied masters in computer science from Georgia Tech", source="data/raw/example.txt", start_char=70, end_char=126),
    ]
    results = retrieve_chunks(query, chunks, top_k=3)
    print(f"test_retrieve_chunks_top_k: query: {query}, results: {results}\n")
    assert len(results) == 1


def test_retrieve_chunks_best_match_first() -> None:
    chunks = [
        Chunk(id="doc-1:: chunk-0", document_id="doc-1", text="Nitesh has experience working with Kafka.", source="data/raw/example.txt", start_char=0, end_char=39),
        Chunk(id="doc-1:: chunk-1", document_id="doc-1", text="He worked in build automation", source="data/raw/example.txt", start_char=40, end_char=69),
        Chunk(id="doc-1:: chunk-2", document_id="doc-1", text="He studied masters in computer science from Georgia Tech", source="data/raw/example.txt", start_char=70, end_char=126),
        Chunk(id="doc-1:: chunk-3", document_id="doc-1", text="Nitesh built developer tooling for platform teams", source="data/raw/example.txt", start_char=126, end_char=175),
    ]
    results = retrieve_chunks("developer tooling", chunks, top_k=1)
    assert results[0][1] == "Nitesh built developer tooling for platform teams"
