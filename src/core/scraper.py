from typing import Optional, Dict, Any
from playwright.async_api import async_playwright
import time
import asyncio
from src.config.settings import ScrapingConfig

class WebScraper:
    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.playwright = None
        self.browser = None
        self.context = None
        self._loop = asyncio.get_event_loop()
        self._loop.run_until_complete(self._initialize_browser())
    
    async def _initialize_browser(self):
        """Initialize the browser and context"""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch()
            self.context = await self.browser.new_context()
        except Exception as e:
            print(f"Error initializing browser: {str(e)}")
            await self.cleanup()
            raise
    
    async def cleanup(self):
        """Clean up browser resources"""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
    
    async def scrape(self, url: str, query: str) -> Optional[Dict[str, Any]]:
        """Scrape content from a URL"""
        if not self.context:
            await self._initialize_browser()
            
        try:
            page = await self.context.new_page()
            await page.goto(url)
            
            # Scroll to bottom and top twice
            for _ in range(2):
                # Scroll to bottom
                await page.evaluate("""
                    () => {
                        window.scrollTo(0, document.body.scrollHeight);
                    }
                """)
                
                # Scroll to top
                await page.evaluate("""
                    () => {
                        window.scrollTo(0, 0);
                    }
                """)
            
            await asyncio.sleep(1)

            content = await page.content()
            
            # Extract text content
            text = await page.evaluate(
                """
                () => {
                    // Remove script and style elements
                    const elements = document.querySelectorAll('script, style');
                    elements.forEach(el => el.remove());
                    
                    // Get text content
                    return document.body.innerText;
                }
            """)
            # Get timestamp before closing page
            timestamp = await page.evaluate("() => new Date().toISOString()")
            
            # Close the page
            await page.close()
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            if len(text) > self.config.dynamic_content_threshold:
                return {
                    "content": text,
                    "metadata": {
                        "url": url,
                        "query": query,
                        "timestamp": timestamp
                    }
                }
            return None
            
        except Exception as e:
            print(f"Error scraping URL {url}: {str(e)}")
            return None
            
    def __del__(self):
        """Cleanup when the object is destroyed"""
        if self._loop.is_running():
            self._loop.create_task(self.cleanup())
        else:
            self._loop.run_until_complete(self.cleanup())