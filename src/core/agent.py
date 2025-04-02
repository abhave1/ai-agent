"""
AI Agent that uses web search, scraping, embeddings, and LLM for comprehensive responses.
"""

from typing import List, Optional
from .search import WebSearch
from .scraper import WebScraper
from .embedding import EmbeddingModel
from .vector_store import VectorStore
from .llm import LLMClient
from ..config.settings import AgentConfig

class AIAgent:
    """AI Agent that combines web search, scraping, embeddings, and LLM."""
    
    def __init__(self, config: AgentConfig = None):
        """Initialize the AI Agent."""
        self.config = config or AgentConfig()
        self.search = WebSearch(self.config.search)
        self.scraper = WebScraper(self.config.scraping)
        self.embedding = EmbeddingModel(self.config.embedding)
        self.vector_store = VectorStore(self.config.vector_store)
        self.llm = LLMClient(self.config.llm)
    
    def process_query(self, query: str) -> Optional[str]:
        """Process a user query and generate a response.
        
        Args:
            query: User query
            
        Returns:
            Generated response if successful
        """
        try:
            # Search for relevant URLs
            urls = self.search.search(query)
            if not urls:
                return self.llm.generate(query)  # Fallback to direct LLM response
            
            # Scrape content from URLs
            contents = []
            for url in urls:
                content = self.scraper.scrape(url)
                if content:
                    contents.append(content)
            
            if not contents:
                return self.llm.generate(query)  # Fallback to direct LLM response
            
            # Generate embeddings for contents
            embeddings = self.embedding.embed_batch(contents)
            if embeddings is not None:
                # Add to vector store
                metadata = [{"text": content} for content in contents]
                self.vector_store.add(embeddings, metadata)
            
            # Generate query embedding
            query_embedding = self.embedding.embed(query)
            if query_embedding is not None:
                # Search for relevant content
                results = self.vector_store.search(query_embedding)
                if results:
                    # Combine relevant texts
                    context = "\n\n".join(metadata["text"] for metadata, _ in results)
                    # Generate response with context
                    return self.llm.generate(f"Context:\n{context}\n\nQuestion: {query}")
            
            # Fallback to direct LLM response
            return self.llm.generate(query)
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return None 