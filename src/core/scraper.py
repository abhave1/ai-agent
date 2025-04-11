"""
Handles web scraping operations using Playwright
"""

from typing import Dict, Any, Optional
from playwright.async_api import async_playwright, Browser, Page
import asyncio
import time
from config.settings import ScrapingConfig

class WebScraper:
    """Handles web scraping operations using Playwright"""
    
    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.browser: Optional[Browser] = None
        self._loop = asyncio.get_event_loop()
        self._playwright = None
        
    async def _initialize_browser(self):
        """Initialize the browser"""
        try:
            self._playwright = await async_playwright().start()
            self.browser = await self._playwright.chromium.launch(
                headless=self.config.headless,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--disable-gpu',
                    '--blink-settings=imagesEnabled=false'
                ]
            )
        except Exception as e:
            print(f"Error initializing browser: {str(e)}")
            if self._playwright:
                await self._playwright.stop()
            raise
        
    async def scrape(self, url: str, topic: str) -> Optional[Dict[str, Any]]:
        """Scrape content from a URL"""
        if not self.browser:
            await self._initialize_browser()
            
        try:
            page = await self.browser.new_page()
            
            # Block image loading
            await page.route("**/*.{png,jpg,jpeg,gif,svg,webp}", lambda route: route.abort())
            
            # Set a longer timeout for page load
            await page.goto(url, timeout=30000)  # 30 seconds timeout
            
            # Wait for the page to be fully loaded
            await page.wait_for_load_state("networkidle")
            
            # Scroll to load dynamic content
            await page.evaluate("""
                () => {
                    window.scrollTo(0, document.body.scrollHeight);
                    return new Promise(resolve => setTimeout(resolve, 2000));
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
        try:
            if self.browser:
                await self.browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            
    def __del__(self):
        """Cleanup when the object is destroyed"""
        if self.browser and self._loop.is_running():
            self._loop.create_task(self.cleanup())