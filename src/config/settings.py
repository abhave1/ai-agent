"""
Configuration settings for the AI Agent.
"""

from typing import Dict, Any
from dataclasses import dataclass, field

@dataclass
class SearchConfig:
    """Search-related configuration."""
    max_results: int = 5
    timeout: int = 30

@dataclass
class ScrapingConfig:
    """Web scraping configuration."""
    timeout: int = 30
    max_retries: int = 3
    wait_for_network_idle: bool = True
    dynamic_content_threshold: int = 1000  # Minimum content length to consider as dynamic

@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    max_sequence_length: int = 512

@dataclass
class VectorStoreConfig:
    """Vector store configuration."""
    index_type: str = "FlatL2"
    similarity_metric: str = "cosine"
    top_k: int = 3

@dataclass
class LLMConfig:
    """LLM configuration."""
    model_name: str = "llama2:13b"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    context_window: int = 4096

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