# imports
from agentic_chatbot_rag_backend import chatbot, get_all_threads, ingest_rag_document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import streamlit as st
import uuid

# extract text
def extract_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    text_parts.append(text)

        return "".join(text_parts)

    return ""

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

    # add_thread(st.session_state["thread_id"])

# load previous conversation
def load_conversation(thread_id):
    state = chatbot.get_state(config = {"configurable": {"thread_id": thread_id}})

    return state.values.get("messages", [])

# get thread title
def get_thread_title(thread_id):

    messages = load_conversation(thread_id)

    for message in messages:

        if isinstance(message, HumanMessage):

            content = extract_text(message.content)

            if content:
                # Keep the sidebar title short
                title = content.strip().replace("\n", " ")

                if len(title) > 35:
                    title = title[:35] + "..."

                return title

    return "New Conversation"

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
# add_thread(st.session_state["thread_id"])

# ======================= Side Bar ===========================

# ============================================================
# PDF UPLOAD FOR RAG (Only one active pdf file at a time)
# ============================================================

with st.sidebar:

    st.subheader("📄 Upload PDF")

    uploaded_file = st.file_uploader(
        "Upload a PDF (Global Knowledge Base)",
        type=["pdf"],
        help="Upload a PDF and then ask questions about its content."
    )

    if uploaded_file is not None:

        # Create temporary file
        temp_pdf_path = "uploaded_document.pdf"

        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())


        # Prevent re-processing the same file
        if (
            "uploaded_pdf_name" not in st.session_state
            or st.session_state["uploaded_pdf_name"]
            != uploaded_file.name
        ):

            with st.spinner("Processing PDF..."):

                try:

                    ingest_rag_document(
                        temp_pdf_path
                    )

                    st.session_state[
                        "uploaded_pdf_name"
                    ] = uploaded_file.name

                    st.success(
                        f"✅ {uploaded_file.name} is ready!"
                    )

                except Exception as e:

                    st.error(
                        f"❌ Failed to process PDF: {str(e)}"
                    )

# sidebar title
st.sidebar.title("History")

# new chat
if st.sidebar.button("New Chat"):
    reset_chat()

    st.rerun()

# different conversations
for thread_id in st.session_state["chat_threads"][::-1]:

    thread_title = get_thread_title(thread_id)

    if st.sidebar.button(thread_title, key=thread_id):
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

            content = extract_text(message.content)

            tmp_mssg.append({"role": role, "content": content})

        st.session_state["message_history"] = tmp_mssg

        st.rerun()

# ======================= Main Chat Interface ======================

# show all messages from the selected conversation
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# user input
user_input = st.chat_input("Type anything here...")

if user_input:
    # Add thread to history only when user actually starts chatting
    add_thread(st.session_state["thread_id"])

    # append user message to history
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.text(user_input)

    # initialize config
    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            # streaming feature
            def response_generator():
                try:
                    for message_chunk, metadata in chatbot.stream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=config,
                        stream_mode="messages"
                    ):
                        # Only stream messages generated by the chatbot node
                        if metadata.get("langgraph_node") != "chat_node":
                            continue

                        content = extract_text(message_chunk.content)

                        if content:
                            yield content

                except Exception as e:
                    yield f"⚠️ Something went wrong: {str(e)}"


            assistant_message = st.write_stream(response_generator())

    # append assistant message to history
    st.session_state["message_history"].append({"role": "assistant", "content": assistant_message})

    # change the ui
    st.rerun()