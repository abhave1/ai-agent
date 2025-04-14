"""
Script to start ChromaDB server.
"""

import uvicorn
from chromadb.config import Settings
from chromadb.server.fastapi import FastAPI

def run_server():
    settings = Settings(
        is_persistent=True,
        persist_directory="../data/chroma",
        allow_reset=True,
        anonymized_telemetry=False
    )
    
    print("Starting ChromaDB server...")
    # Instantiate the FastAPI server
    app_instance = FastAPI(settings)
    
    # Note: In this case, app_instance.app holds the underlying FastAPI app to run.
    try:
        uvicorn.run(
            app_instance.app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nShutting down ChromaDB server...")

if __name__ == "__main__":
    run_server()
