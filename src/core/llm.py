"""
Simple LLM client for text generation.
"""

import requests
from typing import Optional, List, Dict, Union
from src.config.settings import LLMConfig

class LLMClient:
    """Client for interacting with the LLM service."""
    
    def __init__(self, config: LLMConfig):
        """Initialize the LLM client."""
        self.config = config
        self.base_url = config.base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        })
    
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response using the LLM.
        
        Args:
            messages: A list of message dictionaries, e.g., [{"role": "user", "content": "Hello"}]
            
        Returns:
            Generated response text
        """
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "stream": False
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error communicating with Groq API: {str(e)}")
