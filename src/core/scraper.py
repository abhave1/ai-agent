from typing import Optional, Dict, Any
from playwright.sync_api import sync_playwright
import time
from src.config.settings import ScrapingConfig

class WebScraper:
    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch()
        self.context = self.browser.new_context()
    
    def scrape(self, url: str, query: str) -> Optional[Dict[str, Any]]:
        # scrape -> returns Optional[Dict] containing content and metadata
        try:
            page = self.context.new_page()
            page.goto(url)
            
            
            # Scroll to bottom and top twice
            for _ in range(2):
                # Scroll to bottom
                page.evaluate("""
                    () => {
                        window.scrollTo(0, document.body.scrollHeight);
                    }
                """)
                
                # Scroll to top
                page.evaluate("""
                    () => {
                        window.scrollTo(0, 0);
                    }
                """)
            
            time.sleep(1)

            content = page.content()
            
            # Extract text content
            text = page.evaluate(
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
            timestamp = page.evaluate("() => new Date().toISOString()")
            
            # Close the page
            page.close()
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
        self.context.close()
        self.browser.close()
        self.playwright.stop()