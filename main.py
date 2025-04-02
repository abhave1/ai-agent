from langchain_ollama import OllamaLLM
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from pydantic import BaseModel, Field
from prompt import template
from search import google_search

# ✅ Define structured tool using Pydantic
class SearchToolSchema(BaseModel):
    """Search the web for information."""
    query: str = Field(..., description="The search query")

def initialize_model():
    """Initialize the LLM model."""
    return OllamaLLM(model="llama3.2:1b")

def create_chain():
    llm = initialize_model()

    # ✅ Define tool using Tool() wrapper
    search_tool = Tool(
        name="Search",
        func=google_search,
        description="Use this to search the web. Provide a query string.",
        args_schema=SearchToolSchema  # Ensures structured input
    )

    tools = [search_tool]

    # ✅ Create agent using tools
    agent = create_react_agent(llm=llm, tools=tools, prompt=template)
    
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

def main():
    chain = create_chain()
    timeout_seconds = 10

    while True:
        user_input = input("Enter your question (or 'q' to exit): ").strip()
        if user_input.lower() == 'q':
            break
        
        search_query = f"search for {user_input} and provide only the URL results"
        
        # ✅ Invoke properly with expected dict structure
        try:
            response = chain.invoke({"input": search_query})
            print(f"Response:\n{response['output']}")
        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
