"""
Universal MCP Agent - Dynamically discovers and uses any MCP tools with LangGraph ReAct
"""

import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Union, Literal

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from src.config.settings import AgentConfig
from src.core.async_mcp_client import MCPClient
from src.core.llm import LLMClient
from src.core.tool_handler import ToolHandler


class Agent:
    """Universal MCP Agent that dynamically discovers and uses any MCP tools with LangGraph ReAct."""

    def __init__(self, config: AgentConfig = None):
        """Initialize the universal MCP agent with LangGraph ReAct capabilities."""
        # Use default config if none provided
        self.config = config or AgentConfig()

        # Initialize clients using factory methods
        self.mcp_client = self.config.create_mcp_client()
        self.llm_client = self.config.create_llm_client()
        self.tool_handler = ToolHandler(self.mcp_client)

        # Initialize LangGraph LLM using config
        self.llm = ChatGroq(
            model=self.config.llm.model_name,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            groq_api_key=self.config.llm.api_key
        )

        self.connected = False

        # LangGraph specific components
        self.langchain_tools = []
        self.agent_executor = None
        self.memory = MemorySaver()
        self.conversation_context = []  # Enhanced context tracking
        self.session_id = "default_session"  # For session persistence

        # Enhanced system message with dynamic context
        self.base_system_message = '''You are a powerful AI assistant whose primary function is to use tools to answer user requests.

**Core Directive:** You **MUST** use the provided tools whenever possible. Your goal is to execute actions, not to explain how they are done or to write code.

- **NEVER** write code (e.g., Python, JavaScript) or provide code examples to the user.
- **ALWAYS** prefer using a tool over answering directly if a relevant tool exists.
- If a user asks you to perform a task like 'scrape a website', 'get weather', or 'calculate something', you must use the corresponding tool.
- Do not explain the steps you are taking unless the final tool output is an error. Simply provide the result from the tool.
- If the user's request is ambiguous, ask for clarification on the parameters needed for the tool.

Your purpose is to be an action-oriented agent, not a code-writing tutor.'''

    def connect(self) -> bool:
        """Connect to MCP server and initialize LangGraph ReAct agent."""
        if not self.mcp_client.connect():
            return False

        # Discover and build tools using the handler
        if not self.tool_handler.discover_and_build_tools():
            print("❌ Failed to initialize tools using ToolHandler")
            return False

        # Get the tools from the handler
        self.langchain_tools = self.tool_handler.langchain_tools

        # Create LangGraph ReAct agent
        self._create_langgraph_agent()

        self.connected = True
        return True

    def process_message(self, user_input: str, chat_history: List = None) -> str:
        """Process user message with LangGraph ReAct capabilities."""
        if not self.connected:
            return "I'm not connected to the MCP server. Please restart the agent."

        try:
            # Update conversation context
            self.conversation_context.append({"role": "user", "content": user_input})

            # First, determine if we need tools or can respond directly
            if self._should_use_tools(user_input):
                # Use LangGraph ReAct agent for complex queries that need tools
                config = {
                    "configurable": {"thread_id": "main-conversation"},
                    "recursion_limit": 50  # Increase recursion limit
                }

                # Build enhanced messages list for LangGraph
                messages = self._build_enhanced_messages(user_input, chat_history)

                result = self.agent_executor.invoke({"messages": messages}, config)
                response = result["messages"][-1].content

                # Update conversation context
                self.conversation_context.append({"role": "assistant", "content": response})

                return response
            else:
                # Respond directly for simple conversational queries
                response = self._generate_direct_response(user_input, chat_history)
                self.conversation_context.append({"role": "assistant", "content": response})
                return response

        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            self.conversation_context.append({"role": "assistant", "content": error_msg})
            return error_msg

    def _create_langgraph_agent(self):
        """Create LangGraph ReAct agent with industry-standard reasoning."""
        # Use LangGraph's create_react_agent with memory support
        self.agent_executor = create_react_agent(
            model=self.llm,
            tools=self.langchain_tools,
            prompt=self.base_system_message,
            checkpointer=self.memory
        )

    def _should_use_tools(self, user_input: str) -> bool:
        """Determine if the query requires tools."""
        user_lower = user_input.lower()

        # Simple conversational patterns that don't need tools
        simple_patterns = [
            "hi", "hello", "hey", "how are you", "what's up", "good morning",
            "good afternoon", "good evening", "thanks", "thank you", "bye",
            "goodbye", "see you", "nice to meet you", "pleasure", "welcome"
        ]

        # Check for simple greetings/conversation
        for pattern in simple_patterns:
            if pattern in user_lower:
                return False

        # Check for tool-requiring patterns - be more aggressive about tool usage
        tool_patterns = [
            "weather", "temperature", "forecast", "convert", "calculate",
            "scrape", "scraping", "website", "url", "http", "https",
            "location", "coordinates", "geocoding", "geocode",
            "roots", "list", "search", "find", "get data", "current",
            "forecast", "weather", "temperature", "convert"
        ]

        for pattern in tool_patterns:
            if pattern in user_lower:
                return True

        # Check if input contains URLs
        if "http" in user_lower or "www." in user_lower or ".com" in user_lower or ".org" in user_lower:
            return True

        # If it's a question about specific data, likely needs tools
        if "?" in user_input and len(user_input) > 10:
            return True

        # If it's a command-like input, likely needs tools
        if any(word in user_lower for word in ["get", "fetch", "retrieve", "show", "display", "tell me"]):
            return True

        # Default to using tools for most inputs (be more permissive)
        return True

    def _build_enhanced_messages(self, user_input: str, chat_history: List = None) -> List:
        """Build enhanced message list with context and tool information."""
        messages = []

        # Add system message with current context
        system_msg = self._build_dynamic_system_message()
        messages.append(SystemMessage(content=system_msg))

        # Add chat history if provided
        if chat_history:
            messages.extend(chat_history)

        # Add current user input
        messages.append(HumanMessage(content=user_input))

        return messages

    def _build_dynamic_system_message(self) -> str:
        """Build dynamic system message with current context and tool information."""
        base_msg = self.base_system_message

        # Add available tools information
        tool_info = "\n\nAvailable tools:\n"
        # Get tool info from the handler
        for tool_name, tool_schema in self.tool_handler.tools.items():
            description = tool_schema.get("description", "No description available")
            tool_info += f"- {tool_name}: {description}\n"

        # Add recent context if available
        context_info = ""
        if len(self.conversation_context) > 0:
            recent_context = self.conversation_context[-3:]  # Last 3 exchanges
            context_info = "\n\nRecent conversation context:\n"
            for msg in recent_context:
                role = msg["role"].title()
                content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                context_info += f"{role}: {content}\n"

        return base_msg + tool_info + context_info

    def _generate_direct_response(self, user_input: str, chat_history: List = None) -> str:
        """Generate a direct response without tools."""
        # Build conversation context
        messages = []

        # Add chat history if provided
        if chat_history:
            for message in chat_history[-6:]:
                if isinstance(message, HumanMessage):
                    messages.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    messages.append({"role": "assistant", "content": message.content})

        # Add current user input
        messages.append({"role": "user", "content": user_input})

        # Generate response using LLM directly
        response = self.llm.invoke(messages)
        return response.content if hasattr(response, 'content') else str(response)

    def set_session_id(self, session_id: str):
        """Set session ID for conversation persistence."""
        self.session_id = session_id

    def save_conversation_context(self, filepath: str = None):
        """Save conversation context to file for persistence."""
        if filepath is None:
            filepath = f"conversation_{self.session_id}.json"

        try:
            with open(filepath, 'w') as f:
                json.dump({
                    "session_id": self.session_id,
                    "conversation_context": self.conversation_context,
                    "timestamp": str(datetime.now())
                }, f, indent=2)
            print(f"✅ Conversation context saved to {filepath}")
        except Exception as e:
            print(f"⚠️  Failed to save conversation context: {e}")

    def load_conversation_context(self, filepath: str = None):
        """Load conversation context from file."""
        if filepath is None:
            filepath = f"conversation_{self.session_id}.json"

        try:
            import os
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    self.conversation_context = data.get("conversation_context", [])
                    print(f"✅ Conversation context loaded from {filepath}")
                    return True
        except Exception as e:
            print(f"⚠️  Failed to load conversation context: {e}")
        return False

    def close(self):
        """Close MCP connection."""
        if self.connected:
            self.mcp_client.close()
            self.connected = False
            print("✅ Disconnected from MCP server")
