"""
Simple embedding model using sentence-transformers.
"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from ..config.settings import EmbeddingConfig

class EmbeddingModel:
    """Simple embedding model using sentence-transformers."""
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize the embedding model."""
        self.config = config
        self.model = SentenceTransformer(config.model_name)
    
    def embed(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector if successful
        """
        try:
            return self.model.encode(text)
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return None
    
    def embed_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Matrix of embeddings if successful
        """
        try:
            return self.model.encode(texts, batch_size=self.config.batch_size)
        except Exception as e:
            print(f"Error generating batch embeddings: {str(e)}")
            return None 