import json
import os
import logging
import getpass
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field, ValidationError

# Ensure API key is set (prefer environment, fall back to prompt)
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Define state schema
class State(TypedDict):
    messages: Annotated[list, add_messages]

class WeatherInfo(BaseModel):
    percipitation: str = Field(..., description="The amount and type of percipitation.")
    temperature: float = Field(..., description="The temperature in Fahrenheit.")
    wind: str = Field(..., description="The wind speed and direction.")
    condition: str = Field(..., description="The general weather condition (e.g., sunny)")


def determine_weather(location:str, date:str) -> str:
    """
    Use the LLM to return structured weather information for a location and date.

    Args:
        location (str): The location to determine the weather for.
        date (str): The date to determine the weather for.
    """

    messages = [
    (
        "system",
        "You are a helpful assistant that determines the weather for a given location.",
    ),
    ("human", f"What is the weather generally like in {location} on {date}?"),
    ]

    model_with_structure = llm.with_structured_output(WeatherInfo)
    structured_response = model_with_structure.invoke(messages)

    return structured_response


def activity_recommendation(location: str, weather: Dict) -> str:
    """
    Recommend some activities based on the weather and location.
    
    Args:
        location (str): The location.
        weather (Dict): The weather information.
    Returns:
        str: The recommended activity.
    """
    messages = [
    (
        "system",
        "You are a helpful assistant that picks two to three activities based on the weather and location.",
    ),
    ("human", f"What activities would you recommend in {location} on given the weather conditions: \n{json.dumps(weather)}?"),
    ]

    response = llm.invoke(messages)
    return response

# Initialize our state graph
graph_builder = StateGraph(state_schema=State)


# Define our list of tools
tools = [determine_weather, activity_recommendation]

# bind_tools decorates the gemini model so it can emit a structured call to the tools
llm_with_tools = llm.bind_tools(tools=tools, parallel_tool_calls=False)

# Define the system prompt
sys_msg = """
You are a helpful travel assistant that helps with travel recommendations. 
Use the provided tools to determine the weather and recommend activities.
"""

# Define the assistant node function
def assistant(state: State) -> State:
    """
    Assistant node: calls the LLM (with bound tools) and returns updated state.
    """
    return {"messages": [llm_with_tools.invoke(input=[sys_msg] + state["messages"])]}

# Node functions
graph_builder.add_node(node="assistant",action=assistant)
graph_builder.add_node(
    node="tools",
    action=ToolNode(tools=tools),
)

# Edges
graph_builder.add_edge(start_key=START, end_key="assistant")
graph_builder.add_conditional_edges(source = "assistant", path=tools_condition,)
graph_builder.add_edge(start_key="tools", end_key="assistant")

react_graph = graph_builder.compile()

# Add memory
memory = MemorySaver()
react_graph = graph_builder.compile(checkpointer=memory)

# Specify a thread
new_config = {"configurable": {"thread_id": "1"}}

# Run
def main() -> None:
    user_input = input("You: ").strip()

    while user_input != "exit":
        messages = react_graph.invoke(input={"messages": user_input}, config=new_config)
        for m in messages['messages']:
            m.pretty_print()
        user_input = input("You: ")

if __name__ == "__main__":
    main()