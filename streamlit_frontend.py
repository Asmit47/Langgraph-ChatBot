import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from backend_chatbot import chatbot
import uuid



if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = str(uuid.uuid4())

if 'chat_thread' not in st.session_state:
    st.session_state['chat_thread'] = []

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []


def gen_thread():
    thread_id = str(uuid.uuid4())
    return thread_id

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_thread']:
        st.session_state['chat_thread'].append(thread_id)

def reset_chat():
    thread_id = gen_thread()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []

def load_chat(thread_id):
    messages = chatbot.get_state(config={"configurable": {"thread_id": thread_id}}).values.get('messages', [])
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        else:
            role = "assistant"
        result.append({"role": role, "content": msg.content})
    return result

def switch_chat(thread_id):
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = load_chat(thread_id)


st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("History")

for thread in st.session_state['chat_thread']:
    st.sidebar.button(thread,key=thread,on_click=switch_chat,args=(thread,))


CONFIG = {"configurable": {"thread_id": st.session_state['thread_id']}}

for msg in st.session_state['message_history']:
    with st.chat_message(msg["role"]):
        st.text(msg["content"])


if user_input := st.chat_input("Type your message"):
    add_thread(st.session_state['thread_id'])

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