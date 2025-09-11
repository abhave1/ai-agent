"""
Configuration settings for the AI Agent.
"""

import os
from typing import Dict, Any
from dataclasses import dataclass, field

# Load environment variables from .env file
try:
    from dotenv import load_dotenv, dotenv_values
    load_dotenv()
    # Get values from .env file
    env_values = dotenv_values()
except ImportError:
    env_values = {}  # python-dotenv not installed, use empty dict

@dataclass
class SearchConfig:
    """Search-related configuration."""
    max_results: int = 10
    timeout: int = 30

@dataclass
class ScrapingConfig:
    """Web scraping configuration."""
    max_retries: int = 3
    dynamic_content_threshold: int = 1000  # Minimum content length to consider as dynamic
    headless: bool = True  # Whether to run the browser in headless mode
    timeout: int = 5000  # Page load timeout in milliseconds

# @dataclass
# class EmbeddingConfig:
#     """Embedding model configuration."""
#     model_name: str = "all-MiniLM-L6-v2"
#     batch_size: int = 32
#     max_sequence_length: int = 512

@dataclass
class VectorStoreConfig:
    """Vector store configuration."""
    persist_directory: str = "data/chroma"  # Directory to persist ChromaDB data
    collection_name: str = "default"  # Name of the ChromaDB collection
    top_k: int = 3  # Number of results to return in search

@dataclass
class MCPConfig:
    """MCP (Model Context Protocol) configuration."""
    server_script: str = field(default_factory=lambda: env_values.get("MCP_SERVER_SCRIPT", "server.py"))
    command: str = "python3"  # Command to run the server script
    auto_connect: bool = True  # Automatically connect on initialization

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
    """Main agent configuration."""
    search: SearchConfig = field(default_factory=SearchConfig)
    scraping: ScrapingConfig = field(default_factory=ScrapingConfig)
    # embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "search": self.search.__dict__,
            "scraping": self.scraping.__dict__,
            # "embedding": self.embedding.__dict__,
            "vector_store": self.vector_store.__dict__,
            "llm": self.llm.__dict__,
            "mcp": self.mcp.__dict__
        }

# Default configuration instance
DEFAULT_CONFIG = AgentConfig()