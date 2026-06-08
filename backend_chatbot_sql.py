import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import sqlite3


load_dotenv(override=True)

if not os.getenv("NVIDIA_API_KEY"):
    raise ValueError("NVIDIA_API_KEY is not set. Check your .env file!")

llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")

class state(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: state):

    message= state['messages']
    response=llm.invoke(message)
    return {'messages': response}


conn=sqlite3.connect(database="chat_memory.db", check_same_thread=False)

checkpointer = SqliteSaver(conn)

graph = StateGraph(state)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


def fetch_threads():

    thread_ids = checkpointer.list(None)
    unique_thread_ids=set()

    for th in thread_ids:    
        unique_thread_ids.add(th.config['configurable']['thread_id'])

    return unique_thread_ids

