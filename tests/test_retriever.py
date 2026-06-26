from rag_engine.retriever import retrieve_chunks


def test_retreive_chunks_find_matching_chunks() -> None:
    chunks = [
        "Nitesh has experience working with Kafka",
        "He worked in build automation",
        "He studied masters in computer science from Georgia Tech",
    ]
    results = retrieve_chunks("Kafka", chunks, top_k=1)

    assert results[0][1] == "Nitesh has experience working with Kafka"


def test_retrieve_chunks_top_k() -> None:
    query = "Python"
    chunks = ["Python and Django Experience", "Python Automation Work", "Python Scripting"]
    results = retrieve_chunks(query, chunks, top_k=3)
    print(f"test_retrieve_chunks_top_k: query: {query}, results: {results}\n")
    assert len(results) == 3


def test_retrieve_chunks_best_match_first() -> None:
    chunks = [
        "Nitesh built developer platforms",
        "Nitesh built developer tooling for platform teams",
        "Nitesh worked on reliability",
    ]
    results = retrieve_chunks("developer tooling", chunks, top_k=1)
    assert results[0][1] == "Nitesh built developer tooling for platform teams"
