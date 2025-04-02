"""
Vector store module for managing FAISS indices.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import faiss
from ..config.settings import VectorStoreConfig

class VectorStore:
    """Manages FAISS vector store for document embeddings."""
    
    def __init__(self, config: VectorStoreConfig = None):
        """
        Initialize the vector store with configuration.
        
        Args:
            config: Vector store configuration
        """
        self.config = config or VectorStoreConfig()
        self._index = None
        self._metadata = []
    
    def _create_index(self, dimension: int) -> None:
        """
        Create a new FAISS index with the specified dimension.
        
        Args:
            dimension: Dimension of the vectors
        """
        if self.config.index_type == "FlatL2":
            self._index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported index type: {self.config.index_type}")
    
    def add(self, embeddings: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add embeddings and metadata to the vector store.
        
        Args:
            embeddings: Numpy array of embeddings
            metadata: Optional list of metadata dictionaries
        """
        if embeddings is None or len(embeddings) == 0:
            return
        
        # Create index if it doesn't exist
        if self._index is None:
            self._create_index(embeddings.shape[1])
        
        # Add embeddings to the index
        self._index.add(embeddings.astype('float32'))
        
        # Add metadata if provided
        if metadata:
            self._metadata.extend(metadata)
        else:
            # Add empty metadata for each embedding
            self._metadata.extend([{} for _ in range(len(embeddings))])
    
    def search(self, query_embedding: np.ndarray, k: Optional[int] = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar vectors in the index.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return (defaults to config.top_k)
            
        Returns:
            List of tuples containing (metadata, distance) for each result
        """
        if self._index is None:
            return []
        
        k = k or self.config.top_k
        k = min(k, self._index.ntotal)  # Ensure k is not larger than index size
        
        # Search the index
        distances, indices = self._index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k
        )
        
        # Return results with metadata
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self._metadata):  # Ensure valid index
                results.append((self._metadata[idx], float(distance)))
        
        return results
    
    def clear(self) -> None:
        """Clear all data from the vector store."""
        self._index = None
        self._metadata = []
    
    def get_size(self) -> int:
        """
        Get the number of vectors in the store.
        
        Returns:
            int: Number of vectors
        """
        return self._index.ntotal if self._index is not None else 0
    
    def save(self, index_path: str, metadata_path: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            index_path: Path to save the FAISS index
            metadata_path: Path to save the metadata
        """
        if self._index is not None:
            faiss.write_index(self._index, index_path)
            # Save metadata (you might want to use a proper serialization method)
            import json
            with open(metadata_path, 'w') as f:
                json.dump(self._metadata, f)
    
    def load(self, index_path: str, metadata_path: str) -> None:
        """
        Load the vector store from disk.
        
        Args:
            index_path: Path to the FAISS index
            metadata_path: Path to the metadata
        """
        self._index = faiss.read_index(index_path)
        # Load metadata (you might want to use a proper deserialization method)
        import json
        with open(metadata_path, 'r') as f:
            self._metadata = json.load(f) 