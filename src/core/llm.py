"""
Simple LLM client for text generation.
"""

import requests
from typing import Optional, List, Dict, Union, Any
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
    
    def generate_with_tools(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None) -> 'LLMResponse':
        """Generate a response with native function calling support.
        
        Args:
            messages: A list of message dictionaries
            tools: A list of tool definitions in OpenAI format
            
        Returns:
            LLMResponse object with structured tool calls
        """
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "stream": False
        }
        
        # Add tools if provided
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"  # Let LLM decide when to use tools
        
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload
            )
            response.raise_for_status()
            
            # Parse response and return structured object
            response_data = response.json()
            message = response_data["choices"][0]["message"]
            
            return LLMResponse(
                content=message.get("content", ""),
                tool_calls=message.get("tool_calls", []),
                role=message.get("role", "assistant")
            )
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error communicating with Groq API: {str(e)}")


class LLMResponse:
    """Structured response from LLM with tool calling support."""
    
    def __init__(self, content: str, tool_calls: List[Dict[str, Any]], role: str = "assistant"):
        self.content = content
        self.tool_calls = [ToolCall(tc) for tc in tool_calls]
        self.role = role


class ToolCall:
    """Structured tool call from LLM."""
    
    def __init__(self, tool_call_data: Dict[str, Any]):
        self.id = tool_call_data.get("id", "")
        self.type = tool_call_data.get("type", "function")
        self.function = ToolCallFunction(tool_call_data.get("function", {}))


class ToolCallFunction:
    """Function details from tool call."""
    
    def __init__(self, function_data: Dict[str, Any]):
        self.name = function_data.get("name", "")
        self.arguments = function_data.get("arguments", "{}")