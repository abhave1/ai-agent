from typing import Dict, Any, List
from langchain_community.llms import Ollama
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from ..base import BaseAgent, AgentMessage, AgentState
from ...config.settings import DEFAULT_CONFIG
from ...core.search import WebSearch
from ...core.scraper import WebScraper
from ...core.embedding import EmbeddingModel

class BackgroundCollector(BaseAgent):
    """Agent responsible for collecting and processing background data"""
    
    def __init__(self):
        super().__init__("background_collector")
        self.search = WebSearch(DEFAULT_CONFIG.search)
        self.scraper = WebScraper(DEFAULT_CONFIG.scraping)
        self.embedding = EmbeddingModel(DEFAULT_CONFIG.embedding)
        self.llm = Ollama(
            model=DEFAULT_CONFIG.llm.model_name,
            temperature=DEFAULT_CONFIG.llm.temperature,
            num_predict=DEFAULT_CONFIG.llm.max_tokens
        )
        
    def search_and_collect(self, topic: str) -> Dict[str, Any]:
        """Search for relevant URLs and collect their content"""
        # Search for URLs
        urls = self.search.search(topic)
        
        # Scrape content from URLs
        collected_data = []
        for url in urls:
            scraped_content = self.scraper.scrape(url, topic)
            if scraped_content:
                # Generate embeddings for the content
                embedding = self.embedding.embed(scraped_content["content"])
                if embedding is not None:
                    collected_data.append({
                        "url": url,
                        "content": scraped_content["content"],
                        "metadata": scraped_content["metadata"],
                        "embedding": embedding
                    })
        
        return {
            "topic": topic,
            "urls": urls,
            "collected_data": collected_data
        }
        
    def process_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Process incoming messages and trigger appropriate actions"""
        if message.content.get("type") == "collect":
            topic = message.content.get("topic")
            results = self.search_and_collect(topic)
            
            # Store the collected data in the state
            self.state["background_data"][topic] = results
            
            return {
                "status": "success",
                "results": results,
                "topic": topic
            }
        return {"status": "error", "message": "Unknown message type"}
        
    def build_graph(self) -> StateGraph:
        """Build the collection workflow graph"""
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("collect", self.search_and_collect)
        workflow.add_node("store", self.process_message)
        
        # Define edges
        workflow.add_edge(START, "collect")
        workflow.add_edge("collect", "store")
        workflow.add_edge("store", END)
        
        return workflow.compile() 