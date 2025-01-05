import requests
from typing import List, Dict
import json


class OllamaClient:
    def __init__(self, embed_model="mxbai-embed-large", completion_model="llama3.2"):
        self.base_url = "http://localhost:11434"
        self.embed_model = embed_model
        self.completion_model = completion_model

    def get_embedding(self, text: str) -> List[float]:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.embed_model, "prompt": text},
        )
        return response.json()["embedding"]

    def get_streaming_completion(self, messages: List[Dict]):
        """Stream responses from Ollama"""
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={"model": self.completion_model, "messages": messages, "stream": True},
            stream=True,
        )

        for line in response.iter_lines():
            if line:
                json_response = json.loads(line)
                if json_response.get("done", False):
                    break
                yield json_response.get("message", {}).get("content", "")
