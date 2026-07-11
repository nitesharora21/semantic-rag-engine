from rag_engine.retriever import retrieve_chunks


def test_retreive_chunks_find_matching_chunks() -> None:
    chunks = [
        {"id": "chunk-0", "text": "Nitesh has experience working with Kafka"},
        {"id": "chunk-1", "text": "He worked in build automation"},
        {"id": "chunk-2", "text": "He studied masters in computer science from Georgia Tech"},
    ]
    results = retrieve_chunks("Kafka", chunks, top_k=1)
    assert results[0][1] == "Nitesh has experience working with Kafka"


def test_retrieve_chunks_top_k() -> None:
    query = "Python"
    chunks = [
        {"id": "chunk-0", "text": "Python and Django Experience"},
        {"id": "chunk-1", "text": "Python Automation Work"},
        {"id": "chunk-2", "text": "Python Scripting"},
    ]
    results = retrieve_chunks(query, chunks, top_k=3)
    print(f"test_retrieve_chunks_top_k: query: {query}, results: {results}\n")
    assert len(results) == 3


def test_retrieve_chunks_best_match_first() -> None:
    chunks = [
        {"id": "chunk-0", "text": "Nitesh built developer platforms"},
        {"id": "chunk-1", "text": "Nitesh built developer tooling for platform teams"},
        {"id": "chunk-2", "text": "Nitesh worked on reliability"},
    ]
    results = retrieve_chunks("developer tooling", chunks, top_k=1)
    assert results[0][1] == "Nitesh built developer tooling for platform teams"
