from typing import Dict, Any, List
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from .base import BaseAgent, AgentMessage, AgentState
from .background.collector import BackgroundCollector
from .user_facing.retriever import UserFacingRetriever
from ..config.settings import DEFAULT_CONFIG
import asyncio

class AgentOrchestrator:
    """Manages communication and workflow between agents"""
    
    def __init__(self):
        self.agents = {
            "background_collector": BackgroundCollector(),
            "user_facing_retriever": UserFacingRetriever()
        }
        self.state = AgentState(
            messages=[],
            current_agent="orchestrator",
            context={},
            background_data={},
            user_query=""
        )
        self.memory = MemorySaver()
        self._loop = asyncio.get_event_loop()
        
    async def route_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Route message to appropriate agent"""
        if message.receiver in self.agents:
            agent = self.agents[message.receiver]
            return await agent.process_message(message)
        return {"status": "error", "message": f"Unknown agent: {message.receiver}"}
        
    async def schedule_background_collection(self, topic: str) -> Dict[str, Any]:
        """Schedule background collection for a topic"""
        message = AgentMessage(
            sender="orchestrator",
            receiver="background_collector",
            content={"type": "collect", "topic": topic}
        )
        return await self.route_message(message)
        
    async def handle_user_query(self, query: str) -> Dict[str, Any]:
        """Handle user query through the user-facing agent"""
        message = AgentMessage(
            sender="orchestrator",
            receiver="user_facing_retriever",
            content={"type": "query", "query": query}
        )
        return await self.route_message(message)
        
    def build_workflow_graph(self) -> StateGraph:
        """Build the complete workflow graph"""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent's workflow
        for agent_name, agent in self.agents.items():
            agent_graph = agent.build_graph()
            workflow.add_node(agent_name, agent_graph)
            
        # Define edges between agents
        workflow.add_edge("background_collector", "user_facing_retriever")
        workflow.add_edge(START, "background_collector")
        workflow.add_edge("user_facing_retriever", END)
        
        return workflow.compile()
        
    async def run_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete workflow"""
        # Step 1: Background collection
        collection_result = await self.schedule_background_collection(input_data["topic"])
        if collection_result["status"] == "error":
            return collection_result
            
        # Step 2: Handle user query
        query_result = await self.handle_user_query(input_data["query"])
        return query_result 