from rag_engine.models import Chunk
from rag_engine.prompting import build_rag_prompt

def test_build_rag_prompt_includes_question_and_answer_in_context() -> None:
    question = "What Kafka experience does Nitesh have?"
    chunks = [
        Chunk(
            id="resume-profile::chunk-1",
            document_id="resume-profile",
            text="Nitesh is expert in Kafka and has built several event-driven services",
            source="data/raw/resume_profile.txt",
            start_char=300,
            end_char=350,
        )
    ]
    
    # Build the prompt with relevant chunks and user query 
    prompt = build_rag_prompt(question=question, chunks=chunks)
    
    # assertions to ensure the followign
    assert "Nitesh is expert in Kafka and has built several event-driven services" in prompt
    assert "What Kafka experience does Nitesh have?" in prompt
    assert "resume-profile::chunk-1" in prompt
    assert "[Source 1]" in prompt
    
