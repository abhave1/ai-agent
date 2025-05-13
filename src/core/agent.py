"""
AI Agent that uses web search, scraping, embeddings, and LLM for comprehensive responses.
"""

from typing import List, Optional, Dict, Any
from .search import WebSearch
from .scraper import WebScraper
from .embedding import EmbeddingModel
from .vector_store import VectorStore
from .llm import LLMClient
from config.settings import AgentConfig
import asyncio

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
        self._loop = asyncio.get_event_loop()
    
    async def process_query(self, query: str) -> Optional[str]:
        try:
            # urls = self.search.search(query)
            query_embedding = self.embedding.embed(query)
            
            # # Scrape content from URLs in parallel
            # scrape_tasks = [self.scraper.scrape(url, query) for url in urls]
            # scrape_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)
            
            # # Filter out failed scrapes and exceptions
            # contents = []
            # for i, result in enumerate(scrape_results):
            #     if isinstance(result, Exception):
            #         print(f"Error scraping URL {urls[i]}: {str(result)}")
            #         continue
            #     if result:
            #         contents.append(result)
            
            # if not contents:
            #     return None
                    
            # # Generate embeddings for contents
            # content_texts = []
            # content_metadata = []
            # for content in contents:
            #     try:
            #         content_text = content.get("content", "")
            #         if not content_text:
            #             continue
            #         content_texts.append(content_text)
            #         # Add the content text to the metadata
            #         metadata = content.get("metadata", {})
            #         metadata["text"] = content_text  # Add the text to metadata
            #         content_metadata.append(metadata)
            #     except Exception as e:
            #         continue
            
            # if not content_texts:
            #     return None    
            # embeddings = self.embedding.embed_batch(content_texts)
            # if embeddings is not None:
            #     # Add to vector store with full metadata
            #     self.vector_store.add(embeddings, content_metadata)
                
            results = self.vector_store.search(query_embedding)
            if results:
                # Combine relevant texts with their sources
                context_parts = []
                for metadata, _ in results:
                    try:
                        url = metadata.get('url', 'unknown')
                        text = metadata.get('text', '')
                        if not text:
                            print(f"Warning: Empty text in metadata for URL: {url}")
                            continue
                        context_parts.append(f"Source: {url}\nContent: {text}")
                    except Exception as e:
                        print(f"Error processing metadata: {str(e)}")
                        print(f"Metadata object: {metadata}")
                        continue
                
                if not context_parts:
                    print("No valid context parts found, falling back to direct LLM response")
                    return None
                    
                context = "\n\n".join(context_parts)
                # Generate response with context
                return self.llm.generate(f"Context:\n{context}\n\nQuestion: {query}")
                    
            
            # Fallback to direct LLM response
            return None
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            import traceback
            print("Full traceback:")
            print(traceback.format_exc())
            return None 