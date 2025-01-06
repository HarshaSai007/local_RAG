import requests
from typing import List, Dict
import json
import logging


class OllamaClient:
    def __init__(self, embed_model="mxbai-embed-large", completion_model="mistral"):
        self.base_url = "http://localhost:11434"
        self.embed_model = embed_model
        self.completion_model = completion_model
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_embedding(self, text: str) -> List[float]:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.embed_model, "prompt": text},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def get_streaming_completion(self, messages: List[Dict]):
        """Stream responses from Ollama with improved error handling"""
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.completion_model,
                "messages": messages,
                "stream": True,
            },
            stream=True,
            timeout=120,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                try:
                    json_response = json.loads(line)
                    if json_response.get("error"):
                        self.logger.error(f"Ollama error: {json_response['error']}")
                        yield f"Error: {json_response['error']}"
                        break
                    if json_response.get("done", False):
                        break
                    content = json_response.get("message", {}).get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to decode JSON response: {str(e)}")
                    yield "Error: Failed to decode response"
                    break

    def check_server_status(self) -> bool:
        """Check if the Ollama server is running and responsive"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
