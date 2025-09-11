#!/usr/bin/env python3
"""
Sequential Tool Agent - AI can use multiple tools in sequence
Example: Get weather in Celsius → Convert to Fahrenheit → Provide complete answer
"""

import json
import re
import time
import os
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.async_mcp_client import MCPClient
from src.core.llm import LLMClient
from src.config.settings import AgentConfig

class SequentialToolAgent:
    """Agent that can use multiple tools in sequence to complete complex tasks."""
    
    def __init__(self, mcp_server_path: str, llm_config: Dict[str, Any]):
        # Initialize MCP client
        self.mcp_client = MCPClient(mcp_server_path)
        
        # Initialize Groq LLM (using your existing setup)
        from src.config.settings import LLMConfig
        llm_config_obj = LLMConfig(**llm_config)
        self.llm_client = LLMClient(llm_config_obj)
        
        # Initialize tools
        self.tools = {}
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
        
        # Store tools and create descriptions
        self.tools = {}
        self.tool_descriptions = []
        
        for tool_data in tools_result["tools"]:
            try:
                self.tools[tool_data["name"]] = tool_data
                
                # Create tool description for prompt
                params = tool_data.get("input_schema", {}).get("properties", {})
                param_str = ", ".join([f"{k}: {v.get('description', '')}" for k, v in params.items()])
                self.tool_descriptions.append(f"- {tool_data['name']}: {tool_data.get('description', '')} (parameters: {param_str})")
                
                print(f"✅ Added tool: {tool_data['name']}")
            except Exception as e:
                print(f"⚠️  Failed to add tool {tool_data['name']}: {e}")
        
        self.connected = True
        return True
    
    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute MCP tool and return result."""
        try:
            # print(f"🔧 Executing tool '{tool_name}' with arguments: {arguments}")
            
            # Filter out None values
            filtered_args = {k: v for k, v in arguments.items() if v is not None}
            # print(f"🔧 Filtered arguments: {filtered_args}")
            
            # Call MCP tool
            result = self.mcp_client.call_tool(tool_name, filtered_args)
            # print(f"🔧 Raw MCP result: {result}")
            
            if result and "content" in result:
                # Extract text content from MCP response
                content = result["content"]
                if isinstance(content, list) and len(content) > 0:
                    text_result = content[0].get("text", str(result))
                    # print(f"🔧 Extracted text: {text_result}")
                    return text_result
                else:
                    # print(f"🔧 Content is not a list or empty: {content}")
                    return str(result)
            else:
                # print(f"🔧 No content in result: {result}")
                return f"Tool {tool_name} returned no results"
                
        except Exception as e:
            # print(f"❌ Error executing {tool_name}: {str(e)}")
            return f"Error executing {tool_name}: {str(e)}"
    
    def process_message(self, user_input: str, chat_history: List = None) -> str:
        """Process user message with sequential tool usage."""
        if not self.connected:
            return "I'm not connected to the MCP server. Please restart the agent."
        
        # Build tool descriptions
        tools_text = "\n".join(self.tool_descriptions)
        
        # Create conversation context JSON array
        context_json = self._create_context_json(chat_history or [])
        
        # Create sequential tool usage prompt
        sequential_prompt = f"""You are a helpful AI assistant that can use multiple tools in sequence to complete complex tasks.

Available tools:
{tools_text}

**CRITICAL: You MUST use this EXACT format for tool calls:**

For EACH tool you want to use, you MUST include:
Action: tool_name
Action Input: {{"param1": "value1", "param2": "value2"}}

**SEQUENTIAL TOOL USAGE EXAMPLES:**

Example 1 - Weather + Conversion:
Thought: I need to get weather data and convert temperature
Action: weather
Action Input: {{"location": "Phoenix"}}
Action: temperatureConversion
Action Input: {{"temperature": 25, "fromUnit": "celsius", "toUnit": "fahrenheit"}}
Final Answer: The weather in Phoenix is 25°C (77°F) with clear skies.

Example 2 - Location + Weather:
Thought: I need to find coordinates and get weather
Action: geocoding
Action Input: {{"location": "San Francisco"}}
Action: weather
Action Input: {{"location": "San Francisco"}}
Final Answer: San Francisco is at coordinates [lat, lng] and the weather is...

**IMPORTANT RULES:**
1. ALWAYS use the exact format: "Action: tool_name" followed by "Action Input: {{...}}"
2. Use tools when they provide MORE ACCURATE or MORE CURRENT information
3. If you need to process data from one tool with another tool, do so
4. Always end with "Final Answer: [your complete response]"
5. If no tools are needed, respond directly without the Action format

**When to use multiple tools:**
- Weather data + temperature conversion
- Location search + weather for that location
- Any data processing that requires multiple steps

**When NOT to use tools:**
- General knowledge questions you can answer directly
- Simple conversational responses

Previous conversation:
{self._format_chat_history(chat_history or [])}

Human: {user_input}
Assistant:"""

        # Add context JSON array to the prompt
        if context_json and context_json != "[]":
            sequential_prompt += f"\n\nContext: {context_json}"
        
        try:
            # Get response from Groq
            response = self.llm_client.generate([{"role": "user", "content": sequential_prompt}])
            
            # Parse the response for sequential tool calls
            return self._parse_sequential_response(response, user_input)
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def _parse_sequential_response(self, response: str, user_input: str) -> str:
        """Parse response and execute tools sequentially if needed."""
        # print(f"🔍 Parsing LLM response: {response[:200]}...")
        
        # Split response into lines for processing
        lines = response.split('\n')
        
        # Track tool calls and results
        tool_calls = []
        current_tool = None
        current_input = None
        final_answer = None
        
        for line in lines:
            line = line.strip()
            
            # Look for Action: pattern
            if line.startswith('Action:'):
                current_tool = line.replace('Action:', '').strip()
                # print(f"🔍 Found action: {current_tool}")
            
            # Look for Action Input: pattern
            elif line.startswith('Action Input:'):
                input_text = line.replace('Action Input:', '').strip()
                # print(f"🔍 Found action input: {input_text}")
                try:
                    current_input = json.loads(input_text)
                    if current_tool:
                        tool_calls.append((current_tool, current_input))
                        # print(f"✅ Added tool call: {current_tool} with {current_input}")
                except json.JSONDecodeError as e:
                    print(f"⚠️  Could not parse tool input: {input_text} - Error: {e}")
            
            # Look for Final Answer
            elif line.startswith('Final Answer:'):
                final_answer = line.replace('Final Answer:', '').strip()
                # print(f"🔍 Found final answer: {final_answer}")
        
        # Execute tools sequentially if any were found
        if tool_calls:
            # print(f"🔧 Executing {len(tool_calls)} tool(s) sequentially...")
            
            tool_results = []
            for i, (tool_name, tool_input) in enumerate(tool_calls):
                # Validate tool exists
                if tool_name not in self.tools:
                    # print(f"❌ Tool '{tool_name}' not found in available tools: {list(self.tools.keys())}")
                    tool_results.append(f"Step {i+1} ({tool_name}): Error - Tool not found")
                    continue
                
                # print(f"🔧 Step {i+1}: Using {tool_name} with {tool_input}")
                result = self._execute_tool(tool_name, tool_input)
                tool_results.append(f"Step {i+1} ({tool_name}): {result}")
                # print(f"📊 Result: {result}")
            
            # Generate final response with all tool results
            final_prompt = f"""Based on the sequential tool results, provide a complete response to the user.

User question: {user_input}
Tool results:
{chr(10).join(tool_results)}

Provide a natural, complete response that incorporates all the tool results:"""
            
            final_response = self.llm_client.generate([{"role": "user", "content": final_prompt}])
            return final_response
        
        # If we have a final answer but no tool calls, return it
        elif final_answer:
            return final_answer
        
        # No tool calls and no final answer, return the response directly
        # print("🔍 No tool calls or final answer found, returning raw response")
        return response
    
    def _format_chat_history(self, chat_history: List) -> str:
        """Format chat history for the prompt."""
        if not chat_history:
            return "No previous conversation."
        
        formatted = []
        for message in chat_history[-6:]:  # Last 6 messages
            if isinstance(message, HumanMessage):
                formatted.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                formatted.append(f"Assistant: {message.content}")
        
        return "\n".join(formatted)
    
    def _create_context_json(self, chat_history: List) -> str:
        """Create a simple JSON array in OpenAI format."""
        if not chat_history:
            return "[]"
        
        context = []
        for message in chat_history[-10:]:  # Last 10 messages
            if isinstance(message, HumanMessage):
                context.append({
                    "role": "user",
                    "content": message.content
                })
            elif isinstance(message, AIMessage):
                context.append({
                    "role": "assistant",
                    "content": message.content
                })
        
        return json.dumps(context, indent=2)
    
    def save_context(self, chat_history: List, filename: str = "conversation_context.json"):
        """Save conversation context as simple JSON array."""
        try:
            context = []
            for message in chat_history:
                if isinstance(message, HumanMessage):
                    context.append({
                        "role": "user",
                        "content": message.content
                    })
                elif isinstance(message, AIMessage):
                    context.append({
                        "role": "assistant",
                        "content": message.content
                    })
            
            with open(filename, 'w') as f:
                json.dump(context, f, indent=2)
            # print(f"💾 Saved context: {len(context)} messages")
        except Exception as e:
            print(f"⚠️  Failed to save context: {e}")
    
    def load_context(self, filename: str = "conversation_context.json") -> List:
        """Load conversation context from simple JSON array."""
        try:
            if not os.path.exists(filename):
                return []
            
            with open(filename, 'r') as f:
                context = json.load(f)
            
            chat_history = []
            for msg in context:
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history.append(AIMessage(content=msg["content"]))
            
            # print(f"📂 Loaded context: {len(chat_history)} messages")
            return chat_history
        except Exception as e:
            print(f"⚠️  Failed to load context: {e}")
            return []
    
    def close(self):
        """Close MCP connection."""
        if self.connected:
            self.mcp_client.close()
            self.connected = False
            print("✅ Disconnected from MCP server")

def main():
    """Run the sequential tool agent."""
    print("🏢 Sequential Tool Agent - Multi-Step Tool Usage")
    print("=" * 60)
    
    # Configuration
    config = AgentConfig()
    mcp_server_path = "/Users/abhaveabhilash/Documents/Abhave/CodingProjects/i-want/mcp-client/mcp-custom-client/src/everything/dist/index.js"
    
    # Initialize agent
    agent = SequentialToolAgent(mcp_server_path, config.llm.__dict__)
    
    # Connect to MCP server
    if not agent.connect():
        print("❌ Failed to connect to MCP server")
        return
    
    # print(f"✅ Connected! Available tools: {len(agent.tools)}")
    # print("🤖 Sequential Tool Agent ready! Type 'quit' to exit")
    print("=" * 60)
    
    # Initialize empty chat history for session
    chat_history = []
    
    # Chat loop
    while True:
        user_input = input("\n👤 You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        # print("🤖 Thinking...")
        response = agent.process_message(user_input, chat_history)
        print(f"🤖 Bot: {response}")
        
        # Update chat history (kept in memory only)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
    
    agent.close()

if __name__ == "__main__":
    main()
