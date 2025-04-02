"""
Web scraping module for handling both static and dynamic content.
"""

import requests
from typing import Optional, Tuple
from playwright.sync_api import sync_playwright, Page, Browser
from bs4 import BeautifulSoup
from ..config.settings import ScrapingConfig

class WebScraper:
    """Handles web scraping for both static and dynamic content."""
    
    def __init__(self, config: ScrapingConfig = None):
        """Initialize the web scraper with configuration."""
        self.config = config or ScrapingConfig()
        self._playwright = None
        self._browser = None
    
    def _init_playwright(self):
        """Initialize Playwright if not already initialized."""
        if not self._playwright:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=True)
    
    def _cleanup_playwright(self):
        """Clean up Playwright resources."""
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
            self._playwright = None
            self._browser = None
    
    def _is_dynamic_content(self, html_content: str) -> bool:
        """
        Determine if the content is likely dynamic (JavaScript-heavy).
        
        Args:
            html_content: The HTML content to analyze
            
        Returns:
            bool: True if content appears to be dynamic
        """
        # Simple heuristic: check if content length is below threshold
        return len(html_content) < self.config.dynamic_content_threshold
    
    def _scrape_static(self, url: str) -> Optional[str]:
        """
        Scrape static content using requests and BeautifulSoup.
        
        Args:
            url: The URL to scrape
            
        Returns:
            Optional[str]: The HTML content if successful, None otherwise
        """
        try:
            response = requests.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error scraping static content from {url}: {str(e)}")
            return None
    
    def _scrape_dynamic(self, url: str) -> Optional[str]:
        """
        Scrape dynamic content using Playwright.
        
        Args:
            url: The URL to scrape
            
        Returns:
            Optional[str]: The HTML content if successful, None otherwise
        """
        try:
            self._init_playwright()
            page = self._browser.new_page()
            
            # Navigate to the page and wait for network idle
            if self.config.wait_for_network_idle:
                page.goto(url, wait_until='networkidle')
            else:
                page.goto(url)
            
            # Get the page content
            content = page.content()
            page.close()
            return content
            
        except Exception as e:
            print(f"Error scraping dynamic content from {url}: {str(e)}")
            return None
        finally:
            self._cleanup_playwright()
    
    def scrape(self, url: str) -> Optional[str]:
        """
        Scrape content from a URL, automatically choosing the appropriate method.
        
        Args:
            url: The URL to scrape
            
        Returns:
            Optional[str]: The HTML content if successful, None otherwise
        """
        # First try static scraping
        content = self._scrape_static(url)
        
        # If content seems dynamic or static scraping failed, try dynamic scraping
        if not content or self._is_dynamic_content(content):
            content = self._scrape_dynamic(url)
        
        return content
    
    def scrape_with_retry(self, url: str) -> Optional[str]:
        """
        Scrape content with retries.
        
        Args:
            url: The URL to scrape
            
        Returns:
            Optional[str]: The HTML content if successful, None otherwise
        """
        for attempt in range(self.config.max_retries):
            content = self.scrape(url)
            if content:
                return content
            print(f"Attempt {attempt + 1} failed, retrying...")
        return None 