# imports
from agentic_chatbot_backend.agentic_chatbot_backend import chatbot
from langchain_core.messages import BaseMessage, HumanMessage
import streamlit as st

# app title
st.title("Agentic Chatbot With LangGraph")

# initialize config
thread_id = "thread-1"
config = {'configurable': {'thread_id': thread_id}}

# initialize session
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

# user input
user_input = st.chat_input("Type your message here...")

if user_input:
    # append user message to history
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.text(user_input)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            # streaming feature
            assistant = st.write_stream(
            message_chunk.content[0]["text"]
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages"
            )
            if message_chunk.content
            )

    # append assistant message to history
    st.session_state["message_history"].append({"role": "assistant", "content": assistant})