"""
Flask API for vector store operations.
"""

from flask import Flask, request, jsonify
from core.vector_store import VectorStore
from core.embedding import EmbeddingModel
from config.settings import VectorStoreConfig, EmbeddingConfig
import numpy as np

app = Flask(__name__)

# Initialize vector store and embedding model
vector_store = VectorStore(VectorStoreConfig())
embedding_model = EmbeddingModel(EmbeddingConfig())

@app.route('/api/add', methods=['POST'])
def add_to_vector_store():
    """Add text documents to the vector store."""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        metadata = data.get('metadata', [])
        
        if not texts:
            return jsonify({'error': 'No texts provided'}), 400
            
        # Generate embeddings
        embeddings = embedding_model.embed_batch(texts)
        if embeddings is None:
            return jsonify({'error': 'Failed to generate embeddings'}), 500
            
        # Add to vector store
        vector_store.add(embeddings, metadata if metadata else None)
        
        return jsonify({'message': 'Successfully added documents', 'count': len(texts)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search_vector_store():
    """Search for similar documents."""
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
            
        # Generate query embedding
        query_embedding = embedding_model.embed(query)
        if query_embedding is None:
            return jsonify({'error': 'Failed to generate query embedding'}), 500
            
        # Search vector store
        results = vector_store.search(query_embedding)
        
        # Format results
        formatted_results = []
        for metadata, distance in results:
            formatted_results.append({
                'metadata': metadata,
                'distance': distance
            })
            
        return jsonify({
            'results': formatted_results,
            'count': len(formatted_results)
        })
        
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
    app.run(host='0.0.0.0', port=5000) 