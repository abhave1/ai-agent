"""
Vector store implementation using ChromaDB.
"""

import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Tuple
from config.settings import VectorStoreConfig
import os

class VectorStore:
    """Vector store implementation using ChromaDB."""
    
    def __init__(self, config: VectorStoreConfig):
        """Initialize the vector store."""
        self.config = config
        
        # Create persist directory if it doesn't exist
        os.makedirs(config.persist_directory, exist_ok=True)
        
        # Use PersistentClient instead of HttpClient
        self.client = chromadb.PersistentClient(
            path=config.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]] = None) -> None:
        """Add embeddings to the vector store.
        
        Args:
            embeddings: Matrix of embeddings to add
            metadata: Optional metadata for each embedding
        """
        if embeddings is None or len(embeddings) == 0:
            return
            
        # Convert embeddings to list of lists for ChromaDB
        embeddings_list = embeddings.tolist()
        
        # Generate IDs for the embeddings
        ids = [str(i) for i in range(len(embeddings_list))]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings_list,
            ids=ids,
            metadatas=metadata if metadata else None
        )
    
    def search(self, query_embedding: np.ndarray) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            
        Returns:
            List of (metadata, distance) tuples
        """
        # Convert query embedding to list
        query_embedding_list = query_embedding.tolist()
        
        # Get collection size and ensure we request at least 1 result if collection is not empty
        collection_size = self.collection.count()
        if collection_size == 0:
            return []
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding_list],
            n_results=min(self.config.top_k, max(1, collection_size))
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i] if results['metadatas'] else {}
            distance = 1 - results['distances'][0][i]  # Convert cosine similarity to distance
            formatted_results.append((metadata, float(distance)))
        
        return formatted_results
    
    def clear(self) -> None:
        """Clear the vector store."""
        self.client.delete_collection(self.config.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def get_size(self) -> int:
        return self.collection.count()