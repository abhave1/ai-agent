"""
Script to start ChromaDB server.
"""

import chromadb.server.app
import uvicorn

if __name__ == "__main__":
    # Start the ChromaDB server
    app = chromadb.server.app.app
    uvicorn.run(app, host="0.0.0.0", port=8000) 