from typing import Dict, Any, List
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from .base import BaseAgent, AgentMessage, AgentState
from .background.collector import BackgroundCollector
from .user_facing.retriever import UserFacingRetriever
from ..config.settings import DEFAULT_CONFIG

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
        
    def route_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Route message to appropriate agent"""
        if message.receiver in self.agents:
            agent = self.agents[message.receiver]
            return agent.process_message(message)
        return {"status": "error", "message": f"Unknown agent: {message.receiver}"}
        
    def schedule_background_collection(self, topic: str) -> Dict[str, Any]:
        """Schedule background collection for a topic"""
        message = AgentMessage(
            sender="orchestrator",
            receiver="background_collector",
            content={"type": "collect", "topic": topic}
        )
        return self.route_message(message)
        
    def handle_user_query(self, query: str) -> Dict[str, Any]:
        """Handle user query through the user-facing agent"""
        message = AgentMessage(
            sender="orchestrator",
            receiver="user_facing_retriever",
            content={"type": "query", "query": query}
        )
        return self.route_message(message)
        
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
        
    def run_workflow(self, input_data: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the complete workflow"""
        workflow = self.build_workflow_graph()
        if config is None:
            config = {"configurable": {"thread_id": "1"}}
        result = workflow.invoke(input_data, config)
        return result 