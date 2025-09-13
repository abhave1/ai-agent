"""
Tool Handler for Universal MCP Agent - Manages tool discovery, creation, and execution.
"""

import json
import re
import collections.abc
from typing import Dict, Any, List, Optional, Callable, Union, Literal
from pydantic import BaseModel, ValidationError, create_model
from langchain.tools import tool

from src.core.async_mcp_client import MCPClient

class ToolHandler:
    """Manages all tool-related functionality."""

    def __init__(self, mcp_client: MCPClient):
        """Initialize the ToolHandler."""
        self.mcp_client = mcp_client
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.tool_functions: Dict[str, Callable] = {}
        self.langchain_tools: List[Callable] = []
        self.tool_models: Dict[str, BaseModel] = {}
        self.tool_descriptions: List[str] = []

    def discover_and_build_tools(self) -> bool:
        """Discover tools from MCP, build registries, and create LangChain/Pydantic models."""
        tools_result = self.mcp_client.list_tools()
        if not tools_result or "tools" not in tools_result:
            print("❌ Failed to discover tools from MCP server")
            return False

        self._build_dynamic_tool_registry(tools_result["tools"])

        self.langchain_tools = []
        for tool_name, tool_schema in self.tools.items():
            try:
                tool_func = self._create_langchain_tool(tool_name, tool_schema)
                self.langchain_tools.append(tool_func)
            except Exception as e:
                print(f"⚠️  Failed to add LangGraph tool {tool_name}: {e}")

        self._create_pydantic_models()
        return True

    def _build_dynamic_tool_registry(self, tool_schemas: List[Dict[str, Any]]):
        """Build dynamic tool registry from discovered schemas."""
        for tool_schema in tool_schemas:
            try:
                tool_name = tool_schema["name"]
                self.tools[tool_name] = tool_schema
                tool_func = self._make_tool_function(tool_name, tool_schema)
                self.tool_functions[tool_name] = tool_func
                description = self._build_tool_description(tool_schema)
                self.tool_descriptions.append(description)
            except Exception as e:
                print(f"⚠️  Failed to add tool {tool_schema.get('name', 'unknown')}: {e}")

    def _make_tool_function(self, tool_name: str, tool_schema: Dict[str, Any]) -> Callable:
        """Create a dynamic callable function for a tool with a signature matching the schema."""
        input_schema = tool_schema.get("inputSchema", tool_schema.get("input_schema", {}))
        properties = input_schema.get("properties", {})
        required_params = input_schema.get("required", [])
        param_names = list(properties.keys())

        params_list = []
        for name in param_names:
            info = properties.get(name, {})
            param_str = f"{name}: Any"
            if name not in required_params:
                # Extract default value from the schema
                default_value = info.get("default")
                if default_value is not None:
                    param_str += f' = {repr(default_value)}'
                else:
                    param_str += ' = None'
            params_list.append(param_str)

        sig = ", ".join(params_list)
        func_def_lines = [
            f"def {tool_name}({sig}):",
            "    params = locals()",
            f"    return self._execute_tool('{tool_name}', params)"
        ]
        func_def = "\n".join(func_def_lines)

        execution_scope = {'self': self, 'Any': Any}
        exec(func_def, execution_scope)
        tool_func = execution_scope[tool_name]
        tool_func.__doc__ = tool_schema.get("description", f"Dynamically generated tool for {tool_name}.")
        return tool_func

    def _build_tool_description(self, tool_schema: Dict[str, Any]) -> str:
        """Build human-readable tool description for prompts."""
        name = tool_schema["name"]
        description = tool_schema.get("description", f"Tool: {name}")
        description += " Use this tool ONLY for its intended purpose as described."
        
        input_schema = tool_schema.get("inputSchema", tool_schema.get("input_schema", {}))
        properties = input_schema.get("properties", {})
        if properties:
            param_descriptions = []
            required_params = input_schema.get("required", [])
            for param_name, param_info in properties.items():
                param_desc = param_info.get("description", "")
                param_type = param_info.get("type", "string")
                required = param_name in required_params
                required_str = " (required)" if required else " (optional)"
                param_descriptions.append(f"  - {param_name} ({param_type}): {param_desc}{required_str}")
            
            params_text = "\n".join(param_descriptions)
            return f"- {name}: {description}\n{params_text}"
        return f"- {name}: {description}"

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute MCP tool and return result."""
        print(f"\n🌐 MCP CALL: {tool_name}")
        print(f"📨 Arguments: {json.dumps(arguments, indent=2)}")
        
        try:
            if tool_name not in self.tools:
                return f"Error: Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
            
            filtered_args = {k: v for k, v in arguments.items() if v is not None}
            result = self.mcp_client.call_tool(tool_name, filtered_args)
            
            if result and "content" in result:
                content = result["content"]
                if isinstance(content, list) and len(content) > 0:
                    return content[0].get("text", str(result))
                return str(result)
            return f"Tool {tool_name} returned no results"
                
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def get_available_tools(self) -> List[str]:
        return list(self.tools.keys())

    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        return self.tools.get(tool_name)

    def call_tool_by_name(self, tool_name: str, **kwargs) -> str:
        if tool_name not in self.tool_functions:
            return f"Error: Tool '{tool_name}' not found"
        return self.tool_functions[tool_name](**kwargs)

    def _create_langchain_tool(self, tool_name: str, tool_schema: Dict[str, Any]):
        """Create a proper LangChain tool function using @tool decorator."""
        description = self._build_tool_description(tool_schema)
        input_schema = tool_schema.get("inputSchema", tool_schema.get("input_schema", {}))
        properties = input_schema.get("properties", {})
        param_names = list(properties.keys())

        params_list = [f"{name}: Any" for name in param_names]
        sig = ", ".join(params_list)

        func_def_lines = [
            f"def {tool_name}({sig}):",
            "    params = locals()",
            "    input_str = json.dumps(params)",
            f"    return self._execute_mcp_tool_with_params('{tool_name}', input_str)"
        ]
        func_def = "\n".join(func_def_lines)

        execution_scope = {'self': self, 'Any': Any, 'json': json}
        exec(func_def, execution_scope)
        tool_func = execution_scope[tool_name]
        
        tool_func.__name__ = tool_name
        tool_func.__doc__ = description
        return tool(tool_func)

    def _create_pydantic_models(self):
        """Create Pydantic models for each tool's input schema."""
        for tool_name, tool_schema in self.tools.items():
            try:
                input_schema = tool_schema.get("inputSchema", tool_schema.get("input_schema", {}))
                if input_schema:
                    model = self._json_schema_to_pydantic(tool_name, input_schema)
                    self.tool_models[tool_name] = model
            except Exception as e:
                print(f"⚠️  Failed to create Pydantic model for {tool_name}: {e}")

    def _json_schema_to_pydantic(self, tool_name: str, schema: Dict[str, Any]) -> BaseModel:
        """Convert JSON Schema to Pydantic model, with support for enums."""
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        field_definitions = {}
        for prop_name, prop_schema in properties.items():
            field_type = Any
            if "enum" in prop_schema and isinstance(prop_schema["enum"], list) and prop_schema["enum"]:
                if all(isinstance(i, collections.abc.Hashable) for i in prop_schema["enum"]):
                    try:
                        # Use unpacking to create a Literal with multiple values
                        field_type = Literal[*(prop_schema["enum"])]
                    except (TypeError, ValueError):
                        # Broader exception handling for Literal creation
                        field_type = Any
                else:
                    # If any item is not hashable, fallback to Any
                    field_type = Any
            else:
                field_type = self._get_python_type(prop_schema)

            if prop_name in required:
                field_definitions[prop_name] = (field_type, ...)
            else:
                field_definitions[prop_name] = (Optional[field_type], None)
        
        model_name = f"{tool_name.title()}Model"
        return create_model(model_name, **field_definitions)

    def _get_python_type(self, prop_schema: Dict[str, Any]) -> type:
        """Convert JSON Schema type to Python type."""
        prop_type = prop_schema.get("type", "string")
        if prop_type == "string": return str
        elif prop_type == "integer": return int
        elif prop_type == "number": return float
        elif prop_type == "boolean": return bool
        elif prop_type == "array":
            items_schema = prop_schema.get("items")
            # Handle cases where items is missing, an empty dict, or not a dict
            if isinstance(items_schema, dict) and items_schema:
                return List[self._get_python_type(items_schema)]
            return List[Any]
        elif prop_type == "object": return Dict[str, Any]
        else: return Any

    def _coerce_parameter_types(self, params: Dict[str, Any], tool_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce stringified JSON parameters into their correct types based on schema."""
        input_schema = tool_schema.get("inputSchema", tool_schema.get("input_schema", {}))
        properties = input_schema.get("properties", {})
        coerced_params = params.copy()
        
        for param_name, param_value in coerced_params.items():
            if param_name not in properties: continue
            param_schema = properties[param_name]
            param_type = param_schema.get("type")
            
            if (param_type in ["array", "object"]) and isinstance(param_value, str):
                try:
                    coerced_params[param_name] = json.loads(param_value)
                except json.JSONDecodeError:
                    pass
        return coerced_params

    def _execute_mcp_tool_with_params(self, tool_name: str, input_str: str) -> str:
        """Execute MCP tool with enhanced parameter parsing and validation."""
        print(f"\n🔧 TOOL CALL: {tool_name}")
        print(f"📥 Input: {input_str}")
        
        try:
            if tool_name not in self.tools:
                return f"Error: Tool '{tool_name}' not found"
            
            tool_schema = self.tools[tool_name]
            input_schema = tool_schema.get("inputSchema", tool_schema.get("input_schema", {}))
            
            if not input_schema.get("properties"):
                return self._execute_tool(tool_name, {})
            
            try:
                params = json.loads(input_str)
                coerced_params = self._coerce_parameter_types(params, tool_schema)
                
                if tool_name in self.tool_models:
                    validated_params = self._validate_with_pydantic(tool_name, coerced_params)
                    return self._execute_tool(tool_name, validated_params)
                return self._execute_tool(tool_name, coerced_params)
                
            except (json.JSONDecodeError, ValidationError) as e:
                if isinstance(e, ValidationError):
                    return f"Parameter validation error for {tool_name}: {e}"
            
            parsed_params = self._parse_input_against_schema(input_str, tool_schema)
            
            if tool_name in self.tool_models:
                try:
                    validated_params = self._validate_with_pydantic(tool_name, parsed_params)
                    return self._execute_tool(tool_name, validated_params)
                except ValidationError as e:
                    return f"Parameter validation error for {tool_name}: {e}. Please provide valid parameters."
            return self._execute_tool(tool_name, parsed_params)
            
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def _validate_with_pydantic(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters using Pydantic model."""
        model = self.tool_models[tool_name]
        validated = model(**params)
        return validated.model_dump()

    def _parse_input_against_schema(self, input_str: str, tool_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Parse input string against tool schema to extract parameters dynamically."""
        input_schema = tool_schema.get("inputSchema", tool_schema.get("input_schema", {}))
        properties = input_schema.get("properties", {})
        required_params = input_schema.get("required", [])
        
        if len(required_params) == 1 and len(properties) == 1:
            param_name = required_params[0]
            param_type = properties.get(param_name, {}).get("type", "string")
            return {param_name: self._convert_to_type(input_str, param_type)}
        
        parsed_params = {}
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")
            description = param_info.get("description", "").lower()
            
            extracted_value = self._extract_parameter_value(input_str, param_type, description)
            if extracted_value is not None:
                parsed_params[param_name] = extracted_value
        
        if not parsed_params and required_params:
            first_param = required_params[0]
            first_param_type = properties.get(first_param, {}).get("type", "string")
            parsed_params[first_param] = self._convert_to_type(input_str, first_param_type)
        
        return parsed_params

    def _convert_to_type(self, value: str, param_type: str) -> Any:
        """Convert string value to appropriate type based on schema."""
        if param_type == "number":
            try: return float(value)
            except ValueError: return value
        elif param_type == "integer":
            try: return int(value)
            except ValueError: return value
        elif param_type == "boolean":
            return value.lower() in ['true', '1', 'yes', 'on']
        return value

    def _extract_parameter_value(self, input_str: str, param_type: str, description: str) -> Any:
        """Extract parameter value based on type and description patterns."""
        if param_type in ["number", "integer"]:
            m = re.search(r'-?\d+\.?\d*', input_str)
            if m: return self._convert_to_type(m.group(), param_type)
        
        elif param_type == "string":
            if any(k in description for k in ["url", "link", "website"]):
                m = re.search(r'https?://[^\s]+', input_str)
                if m: return m.group()
            
            elif any(k in description for k in ["location", "city", "place"]):
                m = re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', input_str)
                if m: return m.group()
            
            elif "email" in description:
                m = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', input_str)
                if m: return m.group()
            
            return input_str
        
        elif param_type == "boolean":
            if any(w in input_str.lower() for w in ["true", "yes", "on", "enable"]): return True
            elif any(w in input_str.lower() for w in ["false", "no", "off", "disable"]): return False
        
        return None
