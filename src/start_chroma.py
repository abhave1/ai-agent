"""
Script to start ChromaDB server.
"""

from chromadb.config import Settings, ServerSettings

def run_server():
    settings = Settings(
        is_persistent=True,
        persist_directory="data/chroma",
        allow_reset=True,
        anonymized_telemetry=False,
        server=ServerSettings(
            host="0.0.0.0",
            port=8000,
            ssl_enabled=False
        )
    )
    
    print("Starting ChromaDB server...")
    from chromadb.server import Server
    server = Server(settings)
    server.run()

if __name__ == "__main__":
    run_server()