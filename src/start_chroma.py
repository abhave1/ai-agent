"""
Script to start ChromaDB server.
"""

from chromadb.server import FastAPI
import uvicorn
from chromadb.config import Settings

def create_app():
    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="data/chroma",
        allow_reset=True,
        anonymized_telemetry=False
    )
    
    app = FastAPI(settings)
    return app.app

if __name__ == "__main__":
    # Start the ChromaDB server
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)