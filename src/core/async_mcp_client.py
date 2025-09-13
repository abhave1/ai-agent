#!/usr/bin/env python3
"""
MCP client that connects with server
"""

import subprocess
import json
import time
import os
from typing import Dict, Any, Optional
from src.config.settings import MCPConfig

class MCPClient:
    def __init__(self, config: MCPConfig = None):
        self.config = config or MCPConfig()
        self.process = None
        self.request_id = 1
    
    def connect(self) -> bool:
        """Connect to MCP server"""
        try:
            # # Build the command based on config
            # if self.config.command == "python3":
            #     cmd = ["python3", self.config.server_script]
            # elif self.config.command == "node":
            #     cmd = ["node", self.config.server_script]
            # else:
            #     # For custom commands, split by space
            #     cmd = self.config.command.split() + [self.config.server_script]
            cmd = ["node", self.config.server_script]
            
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Initialize using config values
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": self.config.protocol_version,
                    "capabilities": {
                        "roots": {"listChanged": self.config.roots_list_changed}, 
                        "sampling": {} if self.config.sampling_enabled else {},
                        "tools": {"listChanged": self.config.tools_list_changed}
                    },
                    "clientInfo": {"name": self.config.client_name, "version": self.config.client_version}
                }
            }
            
            self.process.stdin.write(json.dumps(init_request) + "\n")
            self.process.stdin.flush()
            
            # Read response (skip any non-JSON output)
            while True:
                response = self.process.stdout.readline()
                if not response:
                    return False
                
                try:
                    data = json.loads(response.strip())
                    if "result" in data:
                        # Send initialized notification
                        initialized_notification = {
                            "jsonrpc": "2.0",
                            "method": "initialized"
                        }
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
    
    def _send_request(self, method: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Send a JSON-RPC request and return the result"""
        if not self.process:
            print("error: No MCP process running")
            return None
        
        try:
            self.request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self.request_id,
                "method": method,
                "params": params or {}
            }
            
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            
            # Read response
            response = self.process.stdout.readline()
            if not response:
                print("error: No response received")
                return None
            
            data = json.loads(response.strip())
            if "error" in data:
                print(f"error: {data['error']}")
                return None
            
            return data.get("result")
        except Exception as e:
            print(f"error: {e}")
            return None
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Call a tool on the MCP server"""
        return self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments or {}
        })
    
    def list_tools(self) -> Optional[Dict[str, Any]]:
        """List available tools from the MCP server"""
        return self._send_request("tools/list")

    def close(self):
        """Close the connection"""
        if self.process:
            self.process.terminate()
            self.process = None