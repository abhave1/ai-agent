#!/usr/bin/env python3
"""
Simple MCP Client - Just what you need to connect to your MCP server
"""

import subprocess
import json
import time
from typing import Dict, Any, Optional

class MCPClient:
    def __init__(self, server_path: str):
        self.server_path = server_path
        self.process = None
        self.request_id = 1
    
    def connect(self) -> bool:
        """Connect to MCP server"""
        try:
            self.process = subprocess.Popen(
                ["node", self.server_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Initialize
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "roots": {"listChanged": True}, 
                        "sampling": {},
                        "tools": {"listChanged": True}
                    },
                    "clientInfo": {"name": "ai-endpoint", "version": "1.0.0"}
                }
            }
            
            print(f"📤 Sending initialize request: {json.dumps(init_request)}")
            self.process.stdin.write(json.dumps(init_request) + "\n")
            self.process.stdin.flush()
            
            # Read response (skip any non-JSON output)
            while True:
                response = self.process.stdout.readline()
                if not response:
                    return False
                
                try:
                    data = json.loads(response.strip())
                    print(f"📥 Initialize response: {data}")
                    if "result" in data:
                        # Send initialized notification
                        initialized_notification = {
                            "jsonrpc": "2.0",
                            "method": "initialized"
                        }
                        print(f"📤 Sending initialized notification: {json.dumps(initialized_notification)}")
                        self.process.stdin.write(json.dumps(initialized_notification) + "\n")
                        self.process.stdin.flush()
                        return True
                    return False
                except json.JSONDecodeError:
                    # Skip non-JSON output (like "undefined")
                    continue
            
            return False
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Call a tool on the MCP server"""
        if not self.process:
            print("❌ No MCP process running")
            return None
        
        try:
            self.request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self.request_id,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments or {}
                }
            }
            
            # print(f"📤 Sending request: {json.dumps(request)}")
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            
            # Read response (skip any non-JSON output)
            max_attempts = 10
            attempt = 0
            while attempt < max_attempts:
                response = self.process.stdout.readline()
                # print(f"📥 Raw response (attempt {attempt + 1}): {repr(response)}")
                
                if not response:
                    # print("❌ No response received")
                    return None
                
                try:
                    data = json.loads(response.strip())
                    # print(f"✅ Parsed response: {data}")
                    if "error" in data:
                        # print(f"❌ Tool call error: {data['error']}")
                        return None
                    return data.get("result")
                except json.JSONDecodeError as e:
                    # print(f"⚠️  Non-JSON output: {response.strip()}")
                    # Skip non-JSON output
                    continue
                
                attempt += 1
            
            # print("❌ Max attempts reached, no valid response")
            return None
        except Exception as e:
            # print(f"❌ Tool call error: {e}")
            return None
    
    def list_tools(self) -> Optional[Dict[str, Any]]:
        """List available tools from the MCP server"""
        if not self.process:
            # print("❌ No MCP process running")
            return None
        
        try:
            self.request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self.request_id,
                "method": "tools/list",
                "params": {}
            }
            
            # print(f"📤 Sending list_tools request: {json.dumps(request)}")
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            
            # Read response
            max_attempts = 10
            attempt = 0
            while attempt < max_attempts:
                response = self.process.stdout.readline()
                # print(f"📥 Raw response (attempt {attempt + 1}): {repr(response)}")
                
                if not response:
                    # print("❌ No response received")
                    return None
                
                try:
                    data = json.loads(response.strip())
                    # print(f"✅ Parsed response: {data}")
                    if "error" in data:
                        # print(f"❌ Tool call error: {data['error']}")
                        return None
                    return data.get("result")
                except json.JSONDecodeError as e:
                    # print(f"⚠️  Non-JSON output: {response.strip()}")
                    continue
                
                attempt += 1
            
            # print("❌ Max attempts reached, no valid response")
            return None
        except Exception as e:
            # print(f"❌ List tools error: {e}")
            return None

    def close(self):
        """Close the connection"""
        if self.process:
            self.process.terminate()
            self.process = None

# Usage in your AI endpoint:
def use_in_your_ai_endpoint():
    """How to use this in your AI endpoint"""
    
    # Connect to your MCP server
    client = MCPClient("/Users/abhaveabhilash/Documents/Abhave/CodingProjects/i-want/mcp-client/mcp-custom-client/src/everything/dist/index.js")
    
    if client.connect():
        # Use MCP tools in your AI logic
        result = client.call_tool("echo", {"message": "Hello from AI!"})
        if result:
            print(f"AI got: {result['content'][0]['text']}")
        
        # Call other tools as needed
        weather = client.call_tool("weather", {"location": "San Francisco"})
        if weather:
            print(f"Weather: {weather['content'][0]['text']}")
        
        client.close()
    else:
        print("Failed to connect to MCP server")

if __name__ == "__main__":
    use_in_your_ai_endpoint()
