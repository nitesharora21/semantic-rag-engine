
import requests

class OllamaGenerator:
    """
    Generate answers using a locally running Ollama Client.
    """
    def __init__(self, model_name: str = "gemma3", base_url: str = "http://localhost:11434") -> None:
        self.model_name = model_name
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
        """
        Generates the response from the given prompt
        """
        response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        },
                    timeout=120,
                )
        response.raise_for_status()

        data = response.json()
        return str(data['response'].strip())
