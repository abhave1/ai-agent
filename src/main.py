"""
Main entry point for the AI Agent application.
"""

import asyncio
from core.agent import AIAgent
from config.settings import DEFAULT_CONFIG

async def process_query(agent: AIAgent, query: str):
    """
    Process a single query and print the response.
    """
    print(f"\nQuery: {query}")
    print("-" * 80)
    
    try:
        response = await agent.process_query(query)
        if response:
            print(f"Response: {response}")
        else:
            print("Failed to generate response.")
    except Exception as e:
        print(f"Error processing query: {str(e)}")
    
    print("-" * 80)

async def main():
    # Initialize the agent
    agent = AIAgent(DEFAULT_CONFIG)
    
    # Process the query
    await process_query(agent, "What is the capital of France?")

if __name__ == "__main__":
    asyncio.run(main()) 