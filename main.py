from src.agents.orchestrator import AgentOrchestrator
from src.config.settings import DEFAULT_CONFIG

def main():
    # Initialize the orchestrator
    orchestrator = AgentOrchestrator()
    
    # Example 1: Schedule background collection
    print("\nExample 1: Scheduling background collection")
    collection_result = orchestrator.schedule_background_collection(
        "latest developments in artificial intelligence"
    )
    print(f"Collection result: {collection_result}")
    
    # Example 2: Handle a user query
    print("\nExample 2: Handling user query")
    query_result = orchestrator.handle_user_query(
        "What are the recent breakthroughs in AI?"
    )
    print(f"Query result: {query_result}")
    
    # Example 3: Run complete workflow
    print("\nExample 3: Running complete workflow")
    workflow_input = {
        "type": "workflow",
        "topic": "quantum computing applications",
        "query": "How is quantum computing being used in real-world applications?"
    }
    workflow_result = orchestrator.run_workflow(workflow_input)
    print(f"Workflow result: {workflow_result}")

if __name__ == "__main__":
    main() 