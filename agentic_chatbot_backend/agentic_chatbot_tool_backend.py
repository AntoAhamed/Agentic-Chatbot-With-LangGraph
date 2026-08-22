# imports
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import os
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
import requests
import math
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch


# load env
load_dotenv()


# define model
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERAPI_API_KEY = os.getenv("WEATHERAPI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", api_key=GEMINI_API_KEY)


# create tools

# search tool
search_tool = TavilySearchResults(max_results=2)

# calculator tool
@tool
def calculator(expression: str) -> str:
    """
    Use this tool ONLY for mathematical calculations.

    The input MUST be a valid mathematical expression.

    Examples:
    - 2 + 2
    - 10 * 5
    - 100 / 4
    - math.sqrt(16)
    - math.pow(2, 3)

    Do NOT use this tool for:
    - current year
    - current date
    - time
    - weather
    - general knowledge
    - web searches
    - questions involving natural language

    If the user asks for current, recent, or up-to-date information,
    use the search tool instead.
    """

    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"

# weather tool
@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    url = (
        f"https://api.weatherapi.com/v1//current.json?"
        f"key={WEATHERAPI_API_KEY}&q={city}"
    )

    response = requests.get(url)

    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temp_c']}°C\n"
        f"Weather: {data['current']['condition']['text']}\n"
        f"Humidity: {data['current']['humidity']}%"
    )


# tools list
tools = [search_tool, calculator, get_weather_data]

llm_with_tools = llm.bind_tools(tools)


# define state
class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]


# chat node
def chat_node(state: ChatState):

    messages = state["messages"]
    
    response = llm_with_tools.invoke(messages)
        
    return {"messages": [response]}


tool_node = ToolNode(tools)


# define checkpoint and connect with database (sqlite)
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)


# define graph & compile
graph = StateGraph(ChatState)


# add nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)


# add edges
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

# compile
chatbot = graph.compile(checkpointer=checkpoint)

# get all threads
def get_all_threads():
    all_threads = set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])

    return list(all_threads)