"""
Simple web search module using DuckDuckGo.
"""

from typing import List
from duckduckgo_search import DDGS
from ..config.settings import SearchConfig

class WebSearch:
    """Simple web search using DuckDuckGo."""
    
    def __init__(self, config: SearchConfig):
        """Initialize the web search."""
        self.config = config
        self.ddgs = DDGS()
    
    def search(self, query: str) -> List[str]:
        """Search for URLs related to the query.
        
        Args:
            query: Search query
            
        Returns:
            List of relevant URLs
        """
        try:
            results = self.ddgs.text(query, max_results=self.config.max_results)
            return [result['link'] for result in results]
        except Exception as e:
            print(f"Error during web search: {str(e)}")
            return [] 