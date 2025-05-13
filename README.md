# AI Agent with RAG and Web Search

An intelligent AI agent that combines web search, content extraction, and RAG (Retrieval Augmented Generation) to provide comprehensive answers to user queries.

## Features

- Web search using DuckDuckGo API
- Intelligent web scraping for both static and dynamic content
- Content extraction and cleaning
- Vector embeddings using SentenceTransformers
- FAISS-based vector storage and retrieval
- LLM-powered response generation using Llama 3.1
- Modular architecture for easy maintenance and extension

## Project Structure

```
src/
├── config/
│   └── settings.py         # Configuration settings
├── core/
│   ├── search.py          # Web search functionality
│   ├── scraper.py         # Web scraping module
│   ├── parser.py          # Content parsing and extraction
│   ├── embeddings.py      # Text embedding generation
│   ├── vector_store.py    # FAISS vector store management
│   └── llm.py             # LLM interaction
├── utils/
│   └── helpers.py         # Utility functions
└── main.py                # Main application entry point
```

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Playwright browsers:
```bash
playwright install
```

4. Set up Ollama and download the Llama 3.1 model:
```bash
ollama pull llama2:13b
```

## Usage

Run the main application:
```bash
python src/main.py
```

## Configuration

The agent can be configured through `src/config/settings.py`. Key settings include:
- Number of search results to fetch
- Scraping timeout settings
- Embedding model configuration
- Vector store parameters
- LLM model settings

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License