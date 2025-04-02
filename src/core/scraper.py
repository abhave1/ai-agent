"""
Simple web scraper using requests and BeautifulSoup.
"""

import requests
from bs4 import BeautifulSoup
from typing import Optional
from ..config.settings import ScrapingConfig

class WebScraper:
    """Simple web scraper using requests and BeautifulSoup."""
    
    def __init__(self, config: ScrapingConfig):
        """Initialize the web scraper."""
        self.config = config
        self.session = requests.Session()
    
    def scrape(self, url: str) -> Optional[str]:
        """Scrape text content from a URL.
        
        Args:
            url: URL to scrape
            
        Returns:
            Extracted text content if successful
        """
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            print(f"Error scraping URL {url}: {str(e)}")
            return None 