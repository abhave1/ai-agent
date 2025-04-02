"""
Main entry point for the AI Agent application.
"""

import os
import argparse
from typing import Optional
from src.config.settings import DEFAULT_CONFIG
from src.core.agent import AIAgent

def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(description="AI Agent with RAG and Web Search")
    parser.add_argument(
        "--query",
        type=str,
        help="The query to process"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--clear-store",
        action="store_true",
        help="Clear the vector store before starting"
    )
    parser.add_argument(
        "--save-store",
        type=str,
        help="Save the vector store to the specified directory"
    )
    parser.add_argument(
        "--load-store",
        type=str,
        help="Load the vector store from the specified directory"
    )
    return parser

def process_query(agent: AIAgent, query: str) -> Optional[str]:
    """
    Process a single query and print the response.
    
    Args:
        agent: The AI agent instance
        query: The query to process
    """
    print(f"\nQuery: {query}")
    print("-" * 80)
    
    response = agent.process_query(query)
    if response:
        print(f"Response: {response}")
    else:
        print("Failed to generate response.")
    
    print("-" * 80)

def interactive_mode(agent: AIAgent):
    """Run the agent in interactive mode."""
    print("AI Agent Interactive Mode")
    print("Type 'exit' to quit")
    print("-" * 80)
    
    while True:
        try:
            query = input("\nEnter your query: ").strip()
            if query.lower() == 'exit':
                break
            if query:
                process_query(agent, query)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")

def main():
    """Main entry point."""
    # Parse command line arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Initialize the agent
    agent = AIAgent(DEFAULT_CONFIG)
    
    # Handle vector store operations
    if args.clear_store:
        agent.clear_vector_store()
        print("Vector store cleared.")
    
    if args.load_store:
        index_path = os.path.join(args.load_store, "index.faiss")
        metadata_path = os.path.join(args.load_store, "metadata.json")
        agent.load_vector_store(index_path, metadata_path)
        print(f"Vector store loaded from {args.load_store}")
    
    # Process query or run interactive mode
    if args.interactive:
        interactive_mode(agent)
    elif args.query:
        process_query(agent, args.query)
    else:
        parser.print_help()
    
    # Save vector store if requested
    if args.save_store:
        os.makedirs(args.save_store, exist_ok=True)
        index_path = os.path.join(args.save_store, "index.faiss")
        metadata_path = os.path.join(args.save_store, "metadata.json")
        agent.save_vector_store(index_path, metadata_path)
        print(f"Vector store saved to {args.save_store}")

if __name__ == "__main__":
    main() 