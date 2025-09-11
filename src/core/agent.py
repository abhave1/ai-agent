"""
AI Agent that uses web search, scraping, embeddings, and LLM for comprehensive responses.
"""

from typing import List, Optional, Dict, Any
from .search import WebSearch
from .scraper import WebScraper
# from .embedding import EmbeddingModel
from .vector_store import VectorStore
from .llm import LLMClient
from .http_mcp_client import HTTPMCPClient, MCPToolCall
from config.settings import AgentConfig
import asyncio
import json

class AIAgent:
    """AI Agent that combines web search, scraping, embeddings, and LLM."""
    
    def __init__(self, mcp_client: HTTPMCPClient, config: AgentConfig = None):
        """Initialize the AI Agent."""
        self.config = config or AgentConfig()
        self.search = WebSearch(self.config.search)
        self.scraper = WebScraper(self.config.scraping)
        # self.embedding = EmbeddingModel(self.config.embedding)
        self.vector_store = VectorStore(self.config.vector_store)
        self.llm = LLMClient(self.config.llm)
        self.mcp_client = mcp_client
        self._loop = asyncio.get_event_loop()
    
    async def process_query(self, query: str) -> Optional[str]:
        try:
            messages = [{"role": "user", "content": query}]

            tools = self.mcp_client.available_tools
            print(f"Tools: {tools}")
            if tools:
                tool_descriptions = []
                for tool in tools.values():
                    tool_descriptions.append(f"- {tool.name}: {tool.description}, schema: {tool.input_schema}")
                    
                    
                print(f"Tool descriptions: {tool_descriptions}")
                
                system_prompt = (
                    "You are a helpful assistant. You have access to the following tools. "
                    "To use a tool, respond with a JSON object with 'tool_name' and 'arguments'.\n"
                    f"Available tools:\n{'\n'.join(tool_descriptions)}"
                )
                messages.insert(0, {"role": "system", "content": system_prompt})

            # For now, we will do a single loop of tool calling. 
            # A more advanced implementation would loop until the LLM doesn't want to call a tool anymore.
            llm_response = self.llm.generate(messages)

            try:
                tool_call_request = json.loads(llm_response)
                if "tool_name" in tool_call_request and "arguments" in tool_call_request:
                    tool_name = tool_call_request["tool_name"]
                    arguments = tool_call_request["arguments"]
                    
                    tool_call = MCPToolCall(name=tool_name, arguments=arguments)
                    tool_result = await self.mcp_client.call_tool(tool_call)

                    messages.append({"role": "assistant", "content": llm_response})
                    messages.append({"role": "user", "content": f"Tool result: {tool_result}"})

                    final_response = self.llm.generate(messages)
                    return final_response
            except (json.JSONDecodeError, TypeError):
                # Not a tool call, just a regular response
                return llm_response

            return llm_response

        except Exception as e:
            print(f"Error processing query: {str(e)}")
            import traceback
            print("Full traceback:")
            print(traceback.format_exc())
            return None 

async def create_agent(mcp_server_url: str, config: AgentConfig = None) -> AIAgent:
    """Create and initialize an AIAgent with an HTTPMCPClient."""
    mcp_client = HTTPMCPClient(mcp_server_url)
    await mcp_client.connect()
    await mcp_client.initialize()
    await mcp_client.discover_tools()
    return AIAgent(mcp_client, config) 