"""
Web search module using DuckDuckGo API.
"""

from typing import List, Dict, Any
from duckduckgo_search import ddg
from ..config.settings import SearchConfig

class WebSearcher:
    """Handles web searches using DuckDuckGo API."""
    
    def __init__(self, config: SearchConfig = None):
        """Initialize the web searcher with configuration."""
        self.config = config or SearchConfig()
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform a web search using DuckDuckGo.
        
        Args:
            query: The search query string
            
        Returns:
            List of dictionaries containing search results with keys:
            - title: The title of the webpage
            - link: The URL of the webpage
            - snippet: A brief description of the content
        """
        try:
            results = ddg(
                query,
                max_results=self.config.max_results,
                timeout=self.config.timeout
            )
            
            # Format results into a consistent structure
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'title': result.get('title', ''),
                    'link': result.get('link', ''),
                    'snippet': result.get('snippet', '')
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error performing web search: {str(e)}")
            return []
    
    def get_urls(self, query: str) -> List[str]:
        """
        Get only the URLs from search results.
        
        Args:
            query: The search query string
            
        Returns:
            List of URLs from search results
        """
        results = self.search(query)
        return [result['link'] for result in results if 'link' in result] 