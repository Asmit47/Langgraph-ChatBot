import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from backend_chatbot_sql import chatbot, fetch_threads, tools
import uuid



if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = str(uuid.uuid4())

if "chat_thread" not in st.session_state:
    st.session_state["chat_thread"] = fetch_threads()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


def gen_thread():
    thread_id = str(uuid.uuid4())
    return thread_id

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_thread']:
        st.session_state['chat_thread'].add(thread_id)

def reset_chat():
    thread_id = gen_thread()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def load_chat(thread_id):
    messages = chatbot.get_state(config={"configurable": {"thread_id": thread_id}}).values.get('messages', [])
    result = []
    pending_tool_info = None
    pending_tool_result = None

    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})

        elif isinstance(msg, AIMessage):
            if msg.tool_calls and not msg.content:
                # Intermediate routing message — capture tool info, don't render yet
                pending_tool_info = {
                    "tool_name": msg.tool_calls[0]["name"],
                    "tool_args": msg.tool_calls[0].get("args", {}),
                }
            elif msg.content:
                # Final response — attach any accumulated tool info and render once
                result.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_info": pending_tool_info,
                    "tool_result": pending_tool_result,
                })
                pending_tool_info = None
                pending_tool_result = None

        elif isinstance(msg, ToolMessage):
            pending_tool_result = msg.content

    return result


def switch_chat(thread_id):
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = load_chat(thread_id)


def render_tool_badge(tool_info: dict | None, tool_result: str | None = None):
    """Render a collapsible tool-call info block."""
    if not tool_info:
        return
    tool_name = tool_info.get("tool_name", "unknown_tool")
    tool_args = tool_info.get("tool_args", {})
    with st.expander(f"🔧 Tool used: **{tool_name}**", expanded=False):
        if tool_args:
            st.markdown("**Query sent to tool:**")
            for k, v in tool_args.items():
                st.code(f"{k}: {v}", language="text")
        if tool_result:
            st.markdown("**Tool result (summary):**")
            # Show a trimmed version so the UI stays clean
            preview = tool_result[:600] + ("…" if len(tool_result) > 600 else "")
            st.caption(preview)


st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("History")
for thread in st.session_state["chat_thread"]:
    st.sidebar.button(
        thread, key=thread, on_click=switch_chat, args=(thread,)
    )


CONFIG = {"configurable": {"thread_id": st.session_state['thread_id']}}

for msg in st.session_state['message_history']:
    with st.chat_message(msg["role"]):
        if msg.get("tool_info"):
            render_tool_badge(msg["tool_info"], msg.get("tool_result"))
        if msg["content"]:
            st.markdown(msg["content"])


if user_input := st.chat_input("Type your message"):
    add_thread(st.session_state['thread_id'])

    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    tool_info_for_display = None
    tool_result_for_display = None

    with st.chat_message("assistant"):
        # Use messages stream mode so we get individual message objects
        stream_gen = chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="messages",
        )

        response_placeholder = st.empty()
        full_response = ""

        tool_badge_shown = False

        for chunk, metadata in stream_gen:
            # Detect AIMessage tool call chunks
            if isinstance(chunk, AIMessage) and chunk.tool_calls and not tool_badge_shown:
                tc = chunk.tool_calls[0]
                tool_info_for_display = {
                    "tool_name": tc["name"],
                    "tool_args": tc.get("args", {}),
                }
                render_tool_badge(tool_info_for_display)
                tool_badge_shown = True

            # Detect ToolMessage (search result)
            elif isinstance(chunk, ToolMessage):
                tool_result_for_display = chunk.content

            # Stream final text content
            elif isinstance(chunk, AIMessage) and chunk.content:
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": full_response,
            "tool_info": tool_info_for_display,
            "tool_result": tool_result_for_display,
        }
    )