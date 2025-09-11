#!/usr/bin/env python3
"""
LangChain Multi-Tool Agent - Sequential Tool Usage
AI can use multiple tools in sequence to complete complex tasks.
"""

import json
import re
from typing import Dict, Any, List, Optional
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.async_mcp_client import MCPClient
from src.config.settings import AgentConfig

class MultiToolLangChainAgent:
    """LangChain agent that can use multiple tools in sequence."""
    
    def __init__(self, mcp_server_path: str, llm_config: Dict[str, Any]):
        # Initialize MCP client
        self.mcp_client = MCPClient(mcp_server_path)
        
        # Initialize Groq LLM using official langchain-groq integration
        self.llm = ChatGroq(
            model=llm_config.get("model", "qwen/qwen3-32b"),
            temperature=llm_config.get("temperature", 0.1),
            max_tokens=llm_config.get("max_tokens", 1024)
        )
        
        # Initialize tools and agent
        self.tools = []
        self.agent_executor = None
        self.connected = False
        
        # Tool descriptions for the prompt
        self.tool_descriptions = []
    
    def connect(self) -> bool:
        """Connect to MCP server and initialize tools."""
        if not self.mcp_client.connect():
            return False
        
        # Discover tools from MCP server
        tools_result = self.mcp_client.list_tools()
        if not tools_result or "tools" not in tools_result:
            print("❌ Failed to discover tools from MCP server")
            return False
        
        # Convert MCP tools to LangChain tools
        self.tools = []
        self.tool_descriptions = []
        
        for tool_data in tools_result["tools"]:
            try:
                # Create LangChain tool
                tool = Tool(
                    name=tool_data["name"],
                    description=tool_data.get("description", f"Tool: {tool_data['name']}"),
                    func=lambda input_str: self._execute_mcp_tool(tool_data["name"], {"input": input_str})
                )
                self.tools.append(tool)
                
                # Create tool description for prompt
                params = tool_data.get("input_schema", {}).get("properties", {})
                param_str = ", ".join([f"{k}: {v.get('description', '')}" for k, v in params.items()])
                self.tool_descriptions.append(f"- {tool_data['name']}: {tool_data.get('description', '')} (parameters: {param_str})")
                
                print(f"✅ Added tool: {tool_data['name']}")
            except Exception as e:
                print(f"⚠️  Failed to add tool {tool_data['name']}: {e}")
        
        # Create LangChain agent with sequential tool usage
        self._create_agent()
        self.connected = True
        return True
    
    def _create_agent(self):
        """Create LangChain agent with sequential tool usage capability."""
        # Initialize the agent using official ChatGroq
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,  # Allow up to 5 tool calls in sequence
            early_stopping_method="generate"  # Stop when we have a good answer
        )
    
    def _execute_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute MCP tool and return result."""
        try:
            # Filter out None values
            filtered_args = {k: v for k, v in arguments.items() if v is not None}
            
            # Call MCP tool
            result = self.mcp_client.call_tool(tool_name, filtered_args)
            
            if result and "content" in result:
                # Extract text content from MCP response
                content = result["content"]
                if isinstance(content, list) and len(content) > 0:
                    return content[0].get("text", str(result))
                else:
                    return str(result)
            else:
                return f"Tool {tool_name} returned no results"
                
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    
    def process_message(self, user_input: str, chat_history: List = None) -> str:
        """Process user message with sequential tool usage."""
        if not self.connected:
            return "I'm not connected to the MCP server. Please restart the agent."
        
        try:
            # Use LangChain agent for sequential tool usage
            result = self.agent_executor.invoke({"input": user_input})
            return result["output"]
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def close(self):
        """Close MCP connection."""
        if self.connected:
            self.mcp_client.close()
            self.connected = False
            print("✅ Disconnected from MCP server")

def main():
    """Run the multi-tool LangChain agent."""
    print("🏢 LangChain Multi-Tool Agent - Sequential Tool Usage")
    print("=" * 60)
    
    # Configuration
    config = AgentConfig()
    mcp_server_path = "/Users/abhaveabhilash/Documents/Abhave/CodingProjects/i-want/mcp-client/mcp-custom-client/src/everything/dist/index.js"
    
    # Initialize agent
    agent = MultiToolLangChainAgent(mcp_server_path, config.llm.__dict__)
    
    # Connect to MCP server
    if not agent.connect():
        print("❌ Failed to connect to MCP server")
        return
    
    print(f"✅ Connected! Available tools: {len(agent.tools)}")
    print("🤖 Multi-Tool Agent ready! Type 'quit' to exit")
    print("=" * 60)
    
    # Chat loop
    chat_history = []
    while True:
        user_input = input("\n👤 You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        print("🤖 Thinking...")
        response = agent.process_message(user_input, chat_history)
        print(f"🤖 Bot: {response}")
        
        # Update chat history
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
    
    agent.close()

if __name__ == "__main__":
    main()
