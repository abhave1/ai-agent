from typing import Annotated, Dict, Any, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

class AgentMessage(BaseModel):
    """Standardized message format for inter-agent communication"""
    sender: str
    receiver: str
    content: Dict[str, Any]
    metadata: Dict[str, Any] = {}

class AgentState(TypedDict):
    """State management for agents"""
    messages: Annotated[List[Dict[str, Any]], add_messages]
    current_agent: str
    context: Dict[str, Any] = {}
    background_data: Dict[str, Any] = {}  # For storing collected data
    user_query: str = ""  # For storing current user query

class BaseAgent:
    """Base class for all agents with communication capabilities"""
    def __init__(self, name: str):
        self.name = name
        self.tools = []
        self.state = AgentState(
            messages=[],
            current_agent=name,
            context={},
            background_data={},
            user_query=""
        )
        
    def send_message(self, receiver: str, content: Dict[str, Any], metadata: Dict[str, Any] = None) -> AgentMessage:
        """Send a message to another agent"""
        message = AgentMessage(
            sender=self.name,
            receiver=receiver,
            content=content,
            metadata=metadata or {}
        )
        self.state["messages"].append(message.dict())
        return message
        
    def receive_message(self) -> List[AgentMessage]:
        """Receive messages for this agent"""
        return [
            AgentMessage(**msg) for msg in self.state["messages"] 
            if msg["receiver"] == self.name
        ]
        
    def process_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Process incoming message and return response"""
        raise NotImplementedError("Subclasses must implement process_message")
        
    def build_graph(self) -> StateGraph:
        """Build the agent's workflow graph"""
        workflow = StateGraph(AgentState)
        return workflow.compile() 