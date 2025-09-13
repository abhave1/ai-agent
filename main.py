#!/usr/bin/env python3
"""
Simple MCP Agent Chatbot
A clean chatbot interface with MCP tool integration using LangGraph ReAct.
"""

import sys
import os
from src.core.agent import Agent
from src.config.settings import AgentConfig

def main():
    """Run the MCP Agent chatbot."""
    print("🤖 MCP Agent Chatbot - Powered by LangGraph ReAct")
    print("=" * 60)
    print("Connecting to MCP server and initializing tools...")
    
    # Initialize agent with default config
    agent = Agent()
    
    # Connect to MCP server and discover tools
    if not agent.connect():
        print("❌ Failed to connect to MCP server")
        print("Make sure your MCP server is running and configured properly.")
        return
    
    print(f"✅ Connected! Available tools: {len(agent.tool_handler.tools)}")
    print(f"✅ LangGraph tools: {len(agent.tool_handler.langchain_tools)}")
    print(f"✅ Pydantic models: {len(agent.tool_handler.tool_models)}")
    print("\n🤖 Chatbot ready! Type 'quit' to exit, 'help' for commands")
    print("=" * 60)
    
    # Load previous conversation if exists
    agent.load_conversation_context()
    
    # Chat loop
    chat_history = []
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                agent.save_conversation_context()
                print("👋 Goodbye! Conversation saved.")
                break
            
            if user_input.lower() in ['help', 'commands']:
                print_help()
                continue
            
            if user_input.lower() in ['save', 'save context']:
                agent.save_conversation_context()
                print("✅ Conversation context saved!")
                continue
            
            if user_input.lower() in ['load', 'load context']:
                agent.load_conversation_context()
                print("✅ Conversation context loaded!")
                continue
            
            if user_input.lower() in ['tools', 'list tools']:
                print_available_tools(agent)
                continue
            
            if user_input.lower() in ['clear', 'reset']:
                chat_history = []
                agent.conversation_context = []
                print("✅ Chat history cleared!")
                continue
            
            if not user_input:
                continue
            
            # Process the message with LangGraph ReAct
            print("🤖 Thinking...")
            response = agent.process_message(user_input, chat_history)
            print(f"🤖 Bot: {response}")
            
            # Update chat history for context
            from langchain_core.messages import HumanMessage, AIMessage
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response))
            
            # Keep chat history manageable (last 20 messages)
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            agent.save_conversation_context()
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    # Clean up
    agent.close()

def print_help():
    """Print available commands."""
    print("\n📋 Available Commands:")
    print("  help, commands     - Show this help message")
    print("  tools, list tools  - List available MCP tools")
    print("  save, save context - Save conversation to file")
    print("  load, load context - Load conversation from file")
    print("  clear, reset       - Clear chat history")
    print("  quit, exit, bye    - Exit the chatbot")
    print("\n💡 Just type your message to chat with the agent!")

def print_available_tools(agent):
    """Print available MCP tools."""
    if not agent.tool_handler.tools:
        print("❌ No tools available")
        return
    
    print(f"\n🔧 Available MCP Tools ({len(agent.tool_handler.tools)}):")
    for tool_name, tool_schema in agent.tool_handler.tools.items():
        description = tool_schema.get("description", "No description")
        print(f"  • {tool_name}: {description}")
    
    print(f"\n✅ LangGraph tools: {len(agent.tool_handler.langchain_tools)}")
    print(f"✅ Pydantic models: {len(agent.tool_handler.tool_models)}")

if __name__ == "__main__":
    main()
