# imports
from agentic_chatbot_backend.agentic_chatbot_db_backend import chatbot, get_all_threads
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import streamlit as st
import uuid

# unique thread id
def generate_thread_id():
    return str(uuid.uuid4())

# add thread
def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

# reset chat
def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()

    st.session_state["message_history"] = []

    add_thread(st.session_state["thread_id"])

# load previous conversation
def load_conversation(thread_id):
    state = chatbot.get_state(config = {"configurable": {"thread_id": thread_id}})

    return state.values.get("messages", [])

# app title
st.title("Agentic Chatbot With LangGraph")

# initialize session thread id
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

# initialize session message history
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# initialize session chat threads
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

# add the current thread id
add_thread(st.session_state["thread_id"])

# ====================== Side Bar =========================

# sidebar title
st.sidebar.title("Agentic Chatbot")

# new chat
if st.sidebar.button("New Chat"):
    reset_chat()

    st.rerun()

# different conversations
for thread_id in st.session_state["chat_threads"][::-1]:

    if st.sidebar.button(str(thread_id), key=thread_id):
        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)

        tmp_mssg = []

        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                continue

            if isinstance(message.content, list):
                content = message.content[0]["text"]
            else:
                content = message.content

            tmp_mssg.append({"role": role, "content": content})

        st.session_state["message_history"] = tmp_mssg

        st.rerun()

# ================== Main Chat Interface ==================

# show all messages from the selected conversation
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

# user input
user_input = st.chat_input("Type anything here...")

if user_input:
    # append user message to history
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.text(user_input)

    # initialize config
    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            # streaming feature
            assistant_message = st.write_stream(
            message_chunk.content[0]["text"]
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages"
            )
            if message_chunk.content
            )

    # append assistant message to history
    st.session_state["message_history"].append({"role": "assistant", "content": assistant_message})