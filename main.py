from agents.orchestrator import AgentOrchestrator
import asyncio

async def main():
    # Initialize the orchestrator
    orchestrator = AgentOrchestrator()
    
    try:
        # # Example 1: Schedule background collection
        # print("\nExample 1: Scheduling background collection\n")
        # collection_result = await orchestrator.schedule_background_collection(
        #     "latest developments in artificial intelligence"
        # )
        # print(f"Collection result: {collection_result}")
        
        # Example 2: Handle user query
        print("\nExample 2: Handling user query\n")
        query_result = await orchestrator.handle_user_query(
            "What are the latest developments in AI?"
        )
        print(f"Query result: {query_result}")
        
        # # Example 3: Run complete workflow
        # print("\nExample 3: Running complete workflow\n")
        # workflow_result = await orchestrator.run_workflow({
        #     "topic": "artificial intelligence",
        #     "query": "What are the latest developments in AI?"
        # })
        # print(f"Workflow result: {workflow_result}")
        
    finally:
        # Cleanup
        del orchestrator

if __name__ == "__main__":
    main()