"""
LLM interaction module using Ollama.
"""

from typing import Optional, Dict, Any
import requests
from ..config.settings import LLMConfig

class LLMInterface:
    """Handles interaction with the Ollama LLM."""
    
    def __init__(self, config: LLMConfig = None):
        """
        Initialize the LLM interface with configuration.
        
        Args:
            config: LLM configuration
        """
        self.config = config or LLMConfig()
        self._api_url = "http://localhost:11434/api/generate"
    
    def _construct_prompt(self, query: str, context: Optional[str] = None) -> str:
        """
        Construct the prompt for the LLM.
        
        Args:
            query: User query
            context: Optional context from retrieved documents
            
        Returns:
            str: Constructed prompt
        """
        if context:
            return f"""Context information is below.
---------------------
{context}
---------------------
Given the context information and no other information, answer the following question:
{query}

Answer:"""
        else:
            return f"""Answer the following question:
{query}

Answer:"""
    
    def _prepare_request(self, prompt: str) -> Dict[str, Any]:
        """
        Prepare the request payload for the Ollama API.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            Dict containing the request parameters
        """
        return {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_tokens
            }
        }
    
    def generate(self, query: str, context: Optional[str] = None) -> Optional[str]:
        """
        Generate a response from the LLM.
        
        Args:
            query: User query
            context: Optional context from retrieved documents
            
        Returns:
            Optional[str]: Generated response if successful, None otherwise
        """
        try:
            # Construct the prompt
            prompt = self._construct_prompt(query, context)
            
            # Prepare the request
            request_data = self._prepare_request(prompt)
            
            # Send request to Ollama API
            response = requests.post(
                self._api_url,
                json=request_data,
                timeout=30
            )
            
            # Check for successful response
            response.raise_for_status()
            
            # Extract and return the generated text
            result = response.json()
            return result.get('response')
            
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama API: {str(e)}")
            return None
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return None
    
    def generate_with_retry(self, query: str, context: Optional[str] = None, max_retries: int = 3) -> Optional[str]:
        """
        Generate a response with retries.
        
        Args:
            query: User query
            context: Optional context from retrieved documents
            max_retries: Maximum number of retry attempts
            
        Returns:
            Optional[str]: Generated response if successful, None otherwise
        """
        for attempt in range(max_retries):
            response = self.generate(query, context)
            if response:
                return response
            print(f"Attempt {attempt + 1} failed, retrying...")
        return None 