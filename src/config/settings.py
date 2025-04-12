"""
Configuration settings for the AI Agent.
"""

from typing import Dict, Any
from dataclasses import dataclass, field

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

@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    max_sequence_length: int = 512

@dataclass
class VectorStoreConfig:
    """Vector store configuration."""
    persist_directory: str = "data/chroma"  # Directory to persist ChromaDB data
    collection_name: str = "default"  # Name of the ChromaDB collection
    top_k: int = 3  # Number of results to return in search

@dataclass
class LLMConfig:
    """LLM configuration."""
    model_name: str = "llama3.2:1b"
    temperature: float = 0.7
    max_tokens: int = 512
    base_url: str = "http://127.0.0.1:11434"

@dataclass
class AgentConfig:
    """Main agent configuration."""
    search: SearchConfig = field(default_factory=SearchConfig)
    scraping: ScrapingConfig = field(default_factory=ScrapingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "search": self.search.__dict__,
            "scraping": self.scraping.__dict__,
            "embedding": self.embedding.__dict__,
            "vector_store": self.vector_store.__dict__,
            "llm": self.llm.__dict__
        }

# Default configuration instance
DEFAULT_CONFIG = AgentConfig() 