import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_nvidia_ai_endpoints import ChatNVIDIA


load_dotenv(override=True)

if not os.getenv("NVIDIA_API_KEY"):
    raise ValueError("NVIDIA_API_KEY is not set. Check your .env file!")


llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")



class state(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


from langchain_core.runnables import RunnableLambda

chat_node = RunnableLambda(lambda state: state["messages"]) | llm | (lambda response: {"messages": response})


checkpointer = InMemorySaver()
graph = StateGraph(state)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


