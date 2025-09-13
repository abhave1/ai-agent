"""
Configuration settings for the AI Agent.
"""

import os
from typing import Dict, Any
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv, dotenv_values
    load_dotenv()
    env_values = dotenv_values()
except ImportError:
    env_values = {}

@dataclass
class MCPConfig:
    """MCP (Model Context Protocol) configuration."""
    server_script: str = field(default_factory=lambda: env_values.get("MCP_SERVER_SCRIPT", "server.py"))
    command: str = field(default_factory=lambda: "node" if env_values.get("MCP_SERVER_SCRIPT", "").endswith(".js") else "python3")
    auto_connect: bool = True  # Automatically connect on initialization
    
    # MCP Protocol Configuration
    protocol_version: str = "2024-11-05"
    client_name: str = "ai-endpoint"
    client_version: str = "1.0.0"
    
    # MCP Capabilities
    roots_list_changed: bool = True
    sampling_enabled: bool = True
    tools_list_changed: bool = True

@dataclass
class LLMConfig:
    """LLM configuration."""
    model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    temperature: float = 0.0  # Set to 1 for more creative responses
    max_tokens: int = 1024  # Reduced for Groq API
    base_url: str = "https://api.groq.com/openai/v1"
    api_key: str = field(default_factory=lambda: env_values.get("GROQ_API_KEY", ""))
    top_p: float = 1.0
    stream: bool = True

@dataclass
class AgentConfig:
    """Agent configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    
    def create_mcp_client(self):
        """Factory method to create MCP client with this config."""
        from src.core.async_mcp_client import MCPClient
        return MCPClient(self.mcp)
    
    def create_llm_client(self):
        """Factory method to create LLM client with this config."""
        from src.core.llm import LLMClient
        return LLMClient(self.llm)

# Default configuration instance
DEFAULT_CONFIG = AgentConfig()