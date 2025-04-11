from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import EmbeddingConfig

class EmbeddingModel:
    """Embedding model using sentence-transformers."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = SentenceTransformer(config.model_name)
    
    def embed(self, text: str) -> Optional[np.ndarray]:
        # embed -> returns Optional[np.ndarray]
        try:
            return self.model.encode(text)
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return None
    
    def embed_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        # embed_batch -> returns Optional[np.ndarray]
        try:
            return self.model.encode(texts, batch_size=self.config.batch_size)
        except Exception as e:
            print(f"Error generating batch embeddings: {str(e)}")
            return None 