from typing import Dict, Any, List
import numpy as np
from langchain_ollama import OllamaLLM
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from ..base import BaseAgent, AgentMessage, AgentState
from ...config.settings import DEFAULT_CONFIG
from ...core.search import WebSearch
from ...core.scraper import WebScraper
from ...core.embedding import EmbeddingModel
import asyncio

class UserFacingRetriever(BaseAgent):
    """Agent responsible for handling user queries and generating responses"""
    
    def __init__(self):
        super().__init__("user_facing_retriever")
        self.search = WebSearch(DEFAULT_CONFIG.search)
        self.scraper = None  # Initialize lazily
        self.embedding = EmbeddingModel(DEFAULT_CONFIG.embedding)
        self.llm = OllamaLLM(
            model=DEFAULT_CONFIG.llm.model_name,
            temperature=DEFAULT_CONFIG.llm.temperature,
            num_predict=DEFAULT_CONFIG.llm.max_tokens
        )
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
    async def _get_scraper(self):
        """Get or initialize the scraper"""
        if self.scraper is None:
            self.scraper = WebScraper(DEFAULT_CONFIG.scraping)
            await self.scraper._initialize_browser()
        return self.scraper
        
    async def find_relevant_content(self, query: str) -> Dict[str, Any]:
        """Find relevant content for a query using semantic search"""
        # Generate embedding for the query
        query_embedding = self.embedding.embed(query)
        if query_embedding is None:
            return {"status": "error", "message": "Failed to generate query embedding"}
            
        # Check if we have background data for this query
        if query in self.state["background_data"]:
            background_data = self.state["background_data"][query]
            # Find most relevant content using embeddings
            relevant_content = []
            for item in background_data["collected_data"]:
                similarity = np.dot(query_embedding, item["embedding"])
                if similarity > 0.7:  # Adjust threshold as needed
                    relevant_content.append(item)
            return {
                "status": "success",
                "content": relevant_content,
                "source": "background_data"
            }
            
        # If no background data, search and collect new content
        urls = self.search.search(query)
        relevant_content = []
        scraper = await self._get_scraper()
        
        for url in urls:
            scraped_content = await scraper.scrape(url, query)
            if scraped_content:
                content_embedding = self.embedding.embed(scraped_content["content"])
                if content_embedding is not None:
                    similarity = np.dot(query_embedding, content_embedding)
                    if similarity > 0.7:  # Adjust threshold as needed
                        relevant_content.append({
                            "url": url,
                            "content": scraped_content["content"],
                            "metadata": scraped_content["metadata"]
                        })
                        
        return {
            "status": "success",
            "content": relevant_content,
            "source": "new_search"
        }
        
    async def process_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Process incoming messages and generate responses"""
        if message.content.get("type") == "query":
            query = message.content.get("query")
            self.state["user_query"] = query
            
            # Find relevant content
            content_result = await self.find_relevant_content(query)
            if content_result["status"] == "error":
                return content_result
                
            # Generate response using LLM
            context = "\n\n".join([item["content"] for item in content_result["content"]])
            response = self.llm.invoke([
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": f"Based on this context: {context}\n\nAnswer this query: {query}"}
            ])
            
            return {
                "status": "success",
                "response": response.content,
                "context": content_result["content"],
                "source": content_result["source"]
            }
        return {"status": "error", "message": "Unknown message type"}
        
    def build_graph(self) -> StateGraph:
        """Build the retrieval workflow graph"""
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("retrieve", self.find_relevant_content)
        workflow.add_node("process", self.process_message)
        
        # Define edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "process")
        workflow.add_edge("process", END)
        
        return workflow.compile()
        
    def __del__(self):
        """Cleanup when the object is destroyed"""
        if self.scraper:
            if self._loop.is_running():
                self._loop.create_task(self.scraper.cleanup())
            else:
                self._loop.run_until_complete(self.scraper.cleanup())
        self._loop.close() 