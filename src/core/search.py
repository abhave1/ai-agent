from typing import List
from duckduckgo_search import DDGS
from config.settings import SearchConfig

"""
WebSearch class
"""
class WebSearch:
    def __init__(self, config: SearchConfig):
        self.config = config # config found in settings.py
        self.ddgs = DDGS()
    
    def search(self, query: str) -> List[str]:
        # search -> returns List[str]
        try:
            results = self.ddgs.text(query, max_results=self.config.max_results)
            urls = []
            for result in results:
                if isinstance(result, dict) and 'href' in result:
                    urls.append(result['href'])
            return urls
        except Exception as e:
            print(f"Error during web search: {str(e)}")
            return []
        
if __name__ == "__main__":
    search = WebSearch(SearchConfig())
    print(search.search("McDonalds Menu Items in the US"))