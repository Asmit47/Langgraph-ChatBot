import os
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.tools import DuckDuckGoSearchRun
import sqlite3


load_dotenv(override=True)

if not os.getenv("NVIDIA_API_KEY"):
    raise ValueError("NVIDIA_API_KEY is not set. Check your .env file!")


# ── Tools ─────────────────────────────────────────────────────────
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

# ── LLM with tools bound ──────────────────────────────────────────────────────
llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")
llm_with_tools = llm.bind_tools(tools)


class state(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: state):
    """Call the LLM (with tools bound). The model decides whether to use a tool."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": response}

def should_use_tool(state: state) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "__end__"

tool_node = ToolNode(tools)

conn=sqlite3.connect(database="chat_memory.db", check_same_thread=False)

checkpointer = SqliteSaver(conn)

graph = StateGraph(state)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", should_use_tool)
graph.add_edge("tools", "chat_node")   # after tool execution, go back to LLM

chatbot = graph.compile(checkpointer=checkpointer)


# ── Helper ────────────────────────────────────────────────────────────────────
def fetch_threads():
    thread_ids = checkpointer.list(None)
    unique_thread_ids = set()
    for th in thread_ids:
        unique_thread_ids.add(th.config["configurable"]["thread_id"])
    return unique_thread_ids
