"""
Flask API for vector store operations and web scraping.
"""

from flask import Flask, request, jsonify
from core.vector_store import VectorStore
from core.embedding import EmbeddingModel
from core.scraper import WebScraper
from core.search import WebSearch
from core.agent import AIAgent
from config.settings import VectorStoreConfig, EmbeddingConfig, ScrapingConfig, SearchConfig, DEFAULT_CONFIG
import numpy as np
import asyncio
from asgiref.sync import async_to_sync

app = Flask(__name__)

# Initialize components
vector_store = VectorStore(VectorStoreConfig())
embedding_model = EmbeddingModel(EmbeddingConfig())
scraper = WebScraper(ScrapingConfig())
search_engine = WebSearch(SearchConfig())
agent = AIAgent(DEFAULT_CONFIG)

@app.route('/api/query', methods=['POST'])
def query_vector_store():
    """Query the vector store with a text query and get structured recipe information."""
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
            
        # Get vector store results first
        query_embedding = embedding_model.embed(query)
        if query_embedding is None:
            return jsonify({'error': 'Failed to generate query embedding'}), 500
            
        results = vector_store.search(query_embedding)
        if not results:
            return jsonify({'error': 'No relevant recipes found'}), 404
            
        # Prepare context for LLM
        context_parts = []
        for metadata, _ in results:
            if metadata.get('text'):
                context_parts.append(metadata['text'])
        
        if not context_parts:
            return jsonify({'error': 'No recipe content found'}), 404
            
        # Create structured prompt for recipe extraction
        prompt = f"""Given the following text content about recipes, extract and list ONLY the recipes mentioned with their ingredients, directions, and nutritional information (if available). Format as bullet points.
DO NOT add any additional commentary, reasoning, or suggestions.
ONLY include information that is explicitly mentioned in the text.

Text content:
{' '.join(context_parts)}

Extract and format as:
- Recipe Name:
  * Ingredients:
    - [list ingredients]
  * Directions:
    - [list steps]
  * Nutrition (if available):
    - [list macros]"""
        
        # Get structured recipe information
        response = agent.llm.generate(prompt)
        
        # Format results with sources
        sources = []
        for metadata, distance in results:
            sources.append({
                'metadata': metadata,
                'relevance_score': 1 - distance
            })
            
        return jsonify({
            'recipes': response,
            'sources': sources,
            'source_count': len(sources)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/scrape', methods=['POST'])
def scrape_and_store():
    """Search the web, scrape content, and store in vector DB."""
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
            
        # Search for URLs
        urls = search_engine.search(query)
        if not urls:
            return jsonify({'error': 'No URLs found for the query'}), 404
            
        # Define the async operation
        async def process_urls():
            contents = []
            for url in urls:
                content = await scraper.scrape(url, query)
                if content:
                    contents.append(content)
            return contents
            
        # Run async operation synchronously
        contents = async_to_sync(process_urls)()
        
        if not contents:
            return jsonify({'error': 'No content could be scraped'}), 404
            
        # Prepare texts and metadata for embedding
        texts = []
        metadata_list = []
        for content in contents:
            text = content.get('content', '')
            if text:
                texts.append(text)
                metadata = content.get('metadata', {})
                metadata['text'] = text
                metadata_list.append(metadata)
        
        # Generate embeddings and store
        if texts:
            embeddings = embedding_model.embed_batch(texts)
            if embeddings is not None:
                vector_store.add(embeddings, metadata_list)
                
                return jsonify({
                    'message': 'Successfully scraped and stored content',
                    'urls_processed': len(urls),
                    'contents_stored': len(texts)
                })
        
        return jsonify({'error': 'Failed to process content'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/count', methods=['GET'])
def get_count():
    """Get the number of documents in the vector store."""
    try:
        count = vector_store.get_size()
        return jsonify({'count': count})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)