"""
Simple vector store using FAISS.
"""

import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from ..config.settings import VectorStoreConfig

class VectorStore:
    """Simple vector store using FAISS."""
    
    def __init__(self, config: VectorStoreConfig):
        """Initialize the vector store."""
        self.config = config
        self.index = None
        self.metadata = []
    
    def add(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]] = None) -> None:
        """Add embeddings to the vector store.
        
        Args:
            embeddings: Matrix of embeddings to add
            metadata: Optional metadata for each embedding
        """
        if embeddings is None or len(embeddings) == 0:
            return
            
        # Initialize index if needed
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Add metadata if provided
        if metadata:
            self.metadata.extend(metadata)
    
    def search(self, query_embedding: np.ndarray) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            
        Returns:
            List of (metadata, distance) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            return []
            
        # Search in index
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            min(self.config.top_k, self.index.ntotal)
        )
        
        # Get results with metadata
        results = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.metadata):
                results.append((self.metadata[idx], float(distance)))
        
        return results
    
    def clear(self) -> None:
        """Clear the vector store."""
        self.index = None
        self.metadata = []
    
    def get_size(self) -> int:
        """
        Get the number of vectors in the store.
        
        Returns:
            int: Number of vectors
        """
        return self.index.ntotal if self.index is not None else 0
    
    def save(self, index_path: str, metadata_path: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            index_path: Path to save the FAISS index
            metadata_path: Path to save the metadata
        """
        if self.index is not None:
            faiss.write_index(self.index, index_path)
            # Save metadata (you might want to use a proper serialization method)
            import json
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f)
    
    def load(self, index_path: str, metadata_path: str) -> None:
        """
        Load the vector store from disk.
        
        Args:
            index_path: Path to the FAISS index
            metadata_path: Path to the metadata
        """
        self.index = faiss.read_index(index_path)
        # Load metadata (you might want to use a proper deserialization method)
        import json
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f) 