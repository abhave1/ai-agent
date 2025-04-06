"""
AI Agent that uses web search, scraping, embeddings, and LLM for comprehensive responses.
"""

from typing import List, Optional, Dict, Any
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
        try:
            # Search for relevant URLs
            urls = self.search.search(query)
            
            # Scrape content from URLs
            contents = []
            for url in urls:
                result = self.scraper.scrape(url, query)
                if result:
                    contents.append(result)
                    
            # Generate embeddings for contents
            embeddings = self.embedding.embed_batch([content["content"] for content in contents])
            if embeddings is not None:
                # Add to vector store with full metadata
                self.vector_store.add(embeddings, [content["metadata"] for content in contents])
            
            # Generate query embedding
            query_embedding = self.embedding.embed(query)
            if query_embedding is not None:
                # Search for relevant content
                results = self.vector_store.search(query_embedding)
                if results:
                    # Combine relevant texts with their sources
                    context_parts = []
                    for metadata, _ in results:
                        context_parts.append(f"Source: {metadata['url']}\nContent: {metadata['text']}")
                    context = "\n\n".join(context_parts)
                    # Generate response with context
                    return self.llm.generate(f"Context:\n{context}\n\nQuestion: {query}")
            
            # Fallback to direct LLM response
            return self.llm.generate(query)
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return None 