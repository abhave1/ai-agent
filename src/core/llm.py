"""
Simple LLM client for text generation.
"""

import requests
import time
from typing import Optional
from ..config.settings import LLMConfig

class LLMClient:
    """Client for interacting with the LLM service."""
    
    def __init__(self, config: LLMConfig):
        """Initialize the LLM client."""
        self.config = config
        self.base_url = "http://localhost:11434"
        self.session = requests.Session()
        self.session.timeout = (30, config.timeout)
    
    def generate(self, prompt: str) -> str:
        """Generate a response using the LLM.
        
        Args:
            prompt: The input prompt
            
        Returns:
            Generated response text
        """
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }
        
        for attempt in range(self.config.retry_attempts):
            try:
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=(30, self.config.timeout)
                )
                response.raise_for_status()
                return response.json()["response"]
                
            except requests.exceptions.Timeout:
                if attempt < self.config.retry_attempts - 1:
                    print(f"Request timed out. Retrying in {self.config.retry_delay} seconds...")
                    time.sleep(self.config.retry_delay)
                    continue
                raise Exception("Request timed out after all retry attempts")
                
            except requests.exceptions.RequestException as e:
                raise Exception(f"Error communicating with Ollama API: {str(e)}")
                
        raise Exception("Failed to generate response after all retry attempts") 