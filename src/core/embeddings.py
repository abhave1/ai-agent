"""
Text embedding generation module using SentenceTransformers.
"""

from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from ..config.settings import EmbeddingConfig

class TextEmbedder:
    """Handles text embedding generation using SentenceTransformers."""
    
    def __init__(self, config: EmbeddingConfig = None):
        """
        Initialize the text embedder with configuration.
        
        Args:
            config: Embedding configuration
        """
        self.config = config or EmbeddingConfig()
        self._model = None
    
    def _load_model(self):
        """Load the SentenceTransformer model if not already loaded."""
        if self._model is None:
            self._model = SentenceTransformer(self.config.model_name)
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before embedding.
        
        Args:
            text: Input text
            
        Returns:
            str: Preprocessed text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Truncate if too long
        if len(text) > self.config.max_sequence_length:
            text = text[:self.config.max_sequence_length]
        return text
    
    def embed(self, text: Union[str, List[str]]) -> Optional[np.ndarray]:
        """
        Generate embeddings for input text.
        
        Args:
            text: Input text or list of texts
            
        Returns:
            Optional[np.ndarray]: Generated embeddings
        """
        try:
            self._load_model()
            
            # Handle single text or list of texts
            if isinstance(text, str):
                text = [self._preprocess_text(text)]
            else:
                text = [self._preprocess_text(t) for t in text]
            
            # Generate embeddings
            embeddings = self._model.encode(
                text,
                batch_size=self.config.batch_size,
                show_progress_bar=False
            )
            
            return embeddings
            
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            return None
    
    def embed_batch(self, texts: List[str], batch_size: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Optional batch size override
            
        Returns:
            Optional[np.ndarray]: Generated embeddings
        """
        try:
            self._load_model()
            
            # Preprocess texts
            processed_texts = [self._preprocess_text(t) for t in texts]
            
            # Use provided batch size or default from config
            current_batch_size = batch_size or self.config.batch_size
            
            # Generate embeddings in batches
            embeddings = self._model.encode(
                processed_texts,
                batch_size=current_batch_size,
                show_progress_bar=True
            )
            
            return embeddings
            
        except Exception as e:
            print(f"Error generating batch embeddings: {str(e)}")
            return None 