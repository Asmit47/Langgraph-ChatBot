import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from backend_chatbot import chatbot

st.title("LangGraph Chatbot")

CONFIG= {"configurable": {"thread_id": "streamlit-session-1"}}

if 'message_history' not in st.session_state:
    st.session_state['message_history']= []

#load session hsitory messages
for msg in st.session_state['message_history']:
    with st.chat_message(msg["role"]):
        st.text(msg["content"])

if user_input := st.chat_input("Type your message..."):
    st.session_state['message_history'].append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.text(user_input)

    with st.chat_message("assistant"):

        response_content = st.write_stream(
            chunk.content for chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            )
        )
        st.session_state['message_history'].append({"role": "assistant", "content": response_content})