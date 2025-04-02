"""
HTML parsing and text extraction module.
"""

from typing import List, Optional
from bs4 import BeautifulSoup
import re

class ContentParser:
    """Handles parsing and cleaning of HTML content."""
    
    def __init__(self):
        """Initialize the content parser."""
        self._unwanted_tags = {
            'script', 'style', 'meta', 'link', 'iframe', 'nav', 'header',
            'footer', 'aside', 'noscript', 'form', 'button', 'input',
            'select', 'textarea', 'label', 'fieldset', 'legend'
        }
        
        self._unwanted_classes = {
            'nav', 'header', 'footer', 'sidebar', 'menu', 'advertisement',
            'ad', 'banner', 'cookie-notice', 'newsletter', 'social-share',
            'comment', 'related-posts', 'pagination'
        }
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing extra whitespace and normalizing.
        
        Args:
            text: The text to clean
            
        Returns:
            str: Cleaned text
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    def _remove_unwanted_elements(self, soup: BeautifulSoup) -> None:
        """
        Remove unwanted HTML elements from the soup object.
        
        Args:
            soup: BeautifulSoup object to clean
        """
        # Remove unwanted tags
        for tag in self._unwanted_tags:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Remove elements with unwanted classes
        for class_name in self._unwanted_classes:
            for element in soup.find_all(class_=re.compile(class_name, re.I)):
                element.decompose()
    
    def _extract_main_content(self, soup: BeautifulSoup) -> Optional[str]:
        """
        Try to identify and extract the main content area.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Optional[str]: Main content text if found, None otherwise
        """
        # Common main content selectors
        main_selectors = [
            'article',
            'main',
            '[role="main"]',
            '#content',
            '.content',
            '#main-content',
            '.main-content',
            '#article-content',
            '.article-content'
        ]
        
        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                return self._clean_text(main_content.get_text())
        
        return None
    
    def parse(self, html_content: str) -> Optional[str]:
        """
        Parse HTML content and extract clean text.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Optional[str]: Cleaned text content if successful, None otherwise
        """
        try:
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            self._remove_unwanted_elements(soup)
            
            # Try to extract main content
            main_content = self._extract_main_content(soup)
            if main_content:
                return main_content
            
            # Fallback to extracting all text if no main content found
            text = soup.get_text(separator=' ', strip=True)
            return self._clean_text(text)
            
        except Exception as e:
            print(f"Error parsing HTML content: {str(e)}")
            return None
    
    def parse_multiple(self, html_contents: List[str]) -> List[str]:
        """
        Parse multiple HTML contents and return list of cleaned texts.
        
        Args:
            html_contents: List of raw HTML contents
            
        Returns:
            List[str]: List of cleaned text contents
        """
        parsed_contents = []
        for html_content in html_contents:
            parsed_content = self.parse(html_content)
            if parsed_content:
                parsed_contents.append(parsed_content)
        return parsed_contents 