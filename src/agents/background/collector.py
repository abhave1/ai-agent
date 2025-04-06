from typing import Dict, Any, List
from langchain_ollama import OllamaLLM
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from ..base import BaseAgent, AgentMessage, AgentState
from ...config.settings import DEFAULT_CONFIG
from ...core.search import WebSearch
from ...core.scraper import WebScraper
from ...core.embedding import EmbeddingModel
import time
import asyncio

class BackgroundCollector(BaseAgent):
    """Agent responsible for collecting and processing background data"""
    
    def __init__(self):
        print("Initializing BackgroundCollector...")
        start_time = time.time()
        super().__init__("background_collector")
        print("Initializing search...")
        self.search = WebSearch(DEFAULT_CONFIG.search)
        self.scraper = None  # Initialize lazily
        print("Initializing embedding model...")
        self.embedding = EmbeddingModel(DEFAULT_CONFIG.embedding)
        print("Initializing LLM...")
        self.llm = OllamaLLM(
            model=DEFAULT_CONFIG.llm.model_name,
            temperature=DEFAULT_CONFIG.llm.temperature,
            num_predict=DEFAULT_CONFIG.llm.max_tokens
        )
        self._loop = asyncio.get_event_loop()
        print(f"BackgroundCollector initialized in {time.time() - start_time:.2f} seconds")
        
    async def _get_scraper(self):
        """Get or initialize the scraper"""
        if self.scraper is None:
            print("Initializing WebScraper...")
            start_time = time.time()
            self.scraper = WebScraper(DEFAULT_CONFIG.scraping)
            await self.scraper._initialize_browser()
            print(f"WebScraper initialized in {time.time() - start_time:.2f} seconds")
        return self.scraper
        
    async def search_and_collect(self, topic: str) -> Dict[str, Any]:
        """Search for relevant URLs and collect their content"""
        print(f"\nStarting search and collection for topic: {topic}")
        start_time = time.time()
        
        try:
            # Search for URLs
            print("Searching for URLs...")
            search_start = time.time()
            urls = self.search.search(topic)
            print(f"Found {len(urls)} URLs in {time.time() - search_start:.2f} seconds")
            
            # Scrape content from URLs
            print("Starting content scraping...")
            collected_data = []
            scraper = await self._get_scraper()
            
            for i, url in enumerate(urls, 1):
                try:
                    print(f"\nProcessing URL {i}/{len(urls)}: {url}")
                    scrape_start = time.time()
                    scraped_content = await scraper.scrape(url, topic)
                    print(f"Scraping completed in {time.time() - scrape_start:.2f} seconds")
                    
                    if scraped_content:
                        print("Generating embedding for content...")
                        embed_start = time.time()
                        embedding = self.embedding.embed(scraped_content["content"])
                        print(f"Embedding generated in {time.time() - embed_start:.2f} seconds")
                        
                        if embedding is not None:
                            collected_data.append({
                                "url": url,
                                "content": scraped_content["content"],
                                "metadata": scraped_content["metadata"],
                                "embedding": embedding
                            })
                            print("Content successfully processed and stored")
                        else:
                            print("Failed to generate embedding for content")
                    else:
                        print("No content scraped from URL")
                except Exception as e:
                    print(f"Error processing URL {url}: {str(e)}")
                    continue
            
            total_time = time.time() - start_time
            print(f"\nSearch and collection completed in {total_time:.2f} seconds")
            print(f"Successfully processed {len(collected_data)}/{len(urls)} URLs")
            
            return {
                "topic": topic,
                "urls": urls,
                "collected_data": collected_data
            }
        except Exception as e:
            print(f"Error in search_and_collect: {str(e)}")
            return {
                "topic": topic,
                "urls": [],
                "collected_data": [],
                "error": str(e)
            }
        
    async def process_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Process incoming messages and trigger appropriate actions"""
        print(f"\nProcessing message of type: {message.content.get('type')}")
        if message.content.get("type") == "collect":
            topic = message.content.get("topic")
            print(f"Starting collection for topic: {topic}")
            results = await self.search_and_collect(topic)
            
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
        print("Building workflow graph...")
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("collect", self.search_and_collect)
        workflow.add_node("store", self.process_message)
        
        # Define edges
        workflow.add_edge(START, "collect")
        workflow.add_edge("collect", "store")
        workflow.add_edge("store", END)
        
        print("Workflow graph built successfully")
        return workflow.compile()
        
    def __del__(self):
        """Cleanup when the object is destroyed"""
        print("Cleaning up BackgroundCollector resources...")
        if self.scraper and self._loop.is_running():
            self._loop.create_task(self.scraper.cleanup()) 