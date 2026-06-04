import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Load environment variables
load_dotenv(override=True)

if not os.getenv("NVIDIA_API_KEY"):
    raise ValueError("NVIDIA_API_KEY is not set. Check your .env file!")

# Initialize the model
llm = ChatNVIDIA(model="moonshotai/kimi-k2.6")

# Define state
class state(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Define node
def chat_node(state: state):
    message = state['messages']
    response = llm.invoke(message)
    return {'messages': response}

# Compile graph
checkpointer = MemorySaver()
graph = StateGraph(state)
graph.add_node(chat_node, 'chat_node')
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)


