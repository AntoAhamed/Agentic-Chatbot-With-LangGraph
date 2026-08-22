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

# load env
load_dotenv()

# define model
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", api_key=GEMINI_API_KEY)

# define state
class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

# chat node
def chat_node(state: ChatState):

    messages = state["messages"]
    
    response = llm.invoke(messages)
        
    return {"messages": [response]}

# define checkpoint and connect with database (sqlite)
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)

# define graph & compile
graph = StateGraph(ChatState)

# add nodes
graph.add_node("chat_node", chat_node)

# add edges
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpoint)

def get_all_threads():
    all_threads = set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])

    return list(all_threads)