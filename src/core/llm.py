"""
Simple LLM client for text generation.
"""

import requests
from typing import Optional
from ..config.settings import LLMConfig

class LLMClient:
    """Client for interacting with the LLM service."""
    
    def __init__(self, config: LLMConfig):
        """Initialize the LLM client."""
        self.config = config
        self.base_url = "http://localhost:11434"
        self.session = requests.Session()
    
    def generate(self, prompt: str) -> str:
        """Generate a response using the LLM.
        
        Args:
            prompt: The input prompt
            
        Returns:
            Generated response text
        """
        payload = {
            "model": "llama3.2:1b",
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            return response.json()["response"]
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error communicating with Ollama API: {str(e)}")
