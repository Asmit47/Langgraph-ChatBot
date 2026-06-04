import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from backend_chatbot import chatbot

st.title("LangGraph Chatbot")

# Initialize thread config
if "config" not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": "streamlit-session-1"}}

# Get current state from memory checkpointer (persistence)
current_state = chatbot.get_state(st.session_state.config)
messages = current_state.values.get("messages", []) if current_state.values else []

# Display conversation history
for msg in messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# Chat input
if user_input := st.chat_input("Type your message..."):
    # Display user message immediately
    with st.chat_message("user"):
        st.write(user_input)
    
    # Invoke LangGraph backend with persistent config
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chatbot.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                st.session_state.config
            )
            
            # Display the assistant's reply (the last message in the graph's state)
            reply_content = response["messages"][-1].content
            st.write(reply_content)