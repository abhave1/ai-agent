"""
Main agent class that orchestrates all components.
"""

from typing import List, Optional, Dict, Any
from .search import WebSearcher
from .scraper import WebScraper
from .parser import ContentParser
from .embeddings import TextEmbedder
from .vector_store import VectorStore
from .llm import LLMInterface
from ..config.settings import AgentConfig

class AIAgent:
    """Main agent class that orchestrates all components."""
    
    def __init__(self, config: AgentConfig = None):
        """
        Initialize the AI agent with configuration.
        
        Args:
            config: Agent configuration
        """
        self.config = config or AgentConfig()
        
        # Initialize components
        self.searcher = WebSearcher(self.config.search)
        self.scraper = WebScraper(self.config.scraping)
        self.parser = ContentParser()
        self.embedder = TextEmbedder(self.config.embedding)
        self.vector_store = VectorStore(self.config.vector_store)
        self.llm = LLMInterface(self.config.llm)
    
    def _process_urls(self, urls: List[str]) -> List[str]:
        """
        Process URLs to extract clean text content.
        
        Args:
            urls: List of URLs to process
            
        Returns:
            List of extracted text contents
        """
        contents = []
        for url in urls:
            # Scrape content
            html_content = self.scraper.scrape_with_retry(url)
            if html_content:
                # Parse content
                text_content = self.parser.parse(html_content)
                if text_content:
                    contents.append(text_content)
        return contents
    
    def _update_vector_store(self, texts: List[str], urls: List[str]) -> None:
        """
        Update the vector store with new texts.
        
        Args:
            texts: List of text contents
            urls: List of corresponding URLs
        """
        if not texts:
            return
        
        # Generate embeddings
        embeddings = self.embedder.embed_batch(texts)
        if embeddings is not None:
            # Create metadata
            metadata = [{"url": url} for url in urls]
            # Add to vector store
            self.vector_store.add(embeddings, metadata)
    
    def _get_relevant_context(self, query: str) -> Optional[str]:
        """
        Get relevant context from the vector store.
        
        Args:
            query: User query
            
        Returns:
            Optional[str]: Relevant context if found
        """
        # Generate query embedding
        query_embedding = self.embedder.embed(query)
        if query_embedding is None:
            return None
        
        # Search vector store
        results = self.vector_store.search(query_embedding)
        if not results:
            return None
        
        # Combine relevant texts
        context_parts = []
        for metadata, _ in results:
            if "text" in metadata:
                context_parts.append(metadata["text"])
        
        return "\n\n".join(context_parts) if context_parts else None
    
    def process_query(self, query: str) -> Optional[str]:
        """
        Process a user query and generate a response.
        
        Args:
            query: User query
            
        Returns:
            Optional[str]: Generated response if successful
        """
        try:
            # Search for relevant URLs
            urls = self.searcher.get_urls(query)
            if not urls:
                return self.llm.generate(query)  # Fallback to direct LLM response
            
            # Process URLs to get content
            contents = self._process_urls(urls)
            
            # Update vector store with new content
            self._update_vector_store(contents, urls)
            
            # Get relevant context
            context = self._get_relevant_context(query)
            
            # Generate response using LLM
            return self.llm.generate_with_retry(query, context)
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return None
    
    def clear_vector_store(self) -> None:
        """Clear the vector store."""
        self.vector_store.clear()
    
    def save_vector_store(self, index_path: str, metadata_path: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            index_path: Path to save the FAISS index
            metadata_path: Path to save the metadata
        """
        self.vector_store.save(index_path, metadata_path)
    
    def load_vector_store(self, index_path: str, metadata_path: str) -> None:
        """
        Load the vector store from disk.
        
        Args:
            index_path: Path to the FAISS index
            metadata_path: Path to the metadata
        """
        self.vector_store.load(index_path, metadata_path) 