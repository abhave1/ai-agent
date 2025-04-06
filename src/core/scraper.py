from typing import Dict, Any, Optional
from playwright.async_api import async_playwright, Browser, Page
import asyncio
import time
from src.config.settings import ScrapingConfig

class WebScraper:
    """Handles web scraping operations using Playwright"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.browser: Optional[Browser] = None
        self._loop = asyncio.get_event_loop()
        
    async def _initialize_browser(self):
        """Initialize the browser"""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(
            headless=self.config.get("headless", True)
        )
        
    async def scrape(self, url: str, topic: str) -> Optional[Dict[str, Any]]:
        """Scrape content from a URL"""
        if not self.browser:
            await self._initialize_browser()
            
        try:
            page = await self.browser.new_page()
            await page.goto(url, timeout=self.config.get("timeout", 30000))
            
            # Scroll to load dynamic content
            await page.evaluate("""
                () => {
                    window.scrollTo(0, document.body.scrollHeight);
                    return new Promise(resolve => setTimeout(resolve, 1000));
                }
            """)
            
            # Extract content
            content = await page.evaluate("""
                () => {
                    const article = document.querySelector('article') || document.body;
                    return {
                        content: article.innerText,
                        metadata: {
                            title: document.title,
                            url: window.location.href
                        }
                    };
                }
            """)
            
            await page.close()
            return content
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None
            
    async def cleanup(self):
        """Cleanup resources"""
        if self.browser:
            await self.browser.close()
            
    def __del__(self):
        """Cleanup when the object is destroyed"""
        if self.browser and self._loop.is_running():
            self._loop.create_task(self.cleanup())