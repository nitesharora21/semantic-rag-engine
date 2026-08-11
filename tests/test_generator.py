from unittest.mock import Mock, patch

from rag_engine.generator import OllamaGenerator

@patch("rag_engine.generator.requests.post")
def test_generate_returns_ollama_response(mock_post: Mock) -> None:
    mock_response = Mock()
    mock_response.json.return_value = {
                "response": "Nitesh built kafka-based event driven services"
            }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    generator = OllamaGenerator(model_name="test_model")
    answer = generator.generate("Test Prompt")

    assert answer == "Nitesh built kafka-based event driven services"

    mock_post.assert_called_once_with(
        "http://localhost:11434/api/generate",
        json={
                "model": "test_model",
                "prompt": "Test Prompt",
                "stream": False,
            },
        timeout=120,
    )
