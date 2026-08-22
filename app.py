# IT INCLUDES HITL FEATURE
# ============================================================
# IMPORTS
# ============================================================

from agentic_chatbot_backend.agentic_chatbot_hitl_backend import chatbot, get_all_threads, ingest_rag_document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.types import Command
import streamlit as st
import uuid


# ============================================================
# EXTRACT TEXT
# ============================================================

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


# ============================================================
# UNIQUE THREAD ID
# ============================================================

def generate_thread_id():
    return str(uuid.uuid4())


# ============================================================
# ADD THREAD
# ============================================================

def add_thread(thread_id):

    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


# ============================================================
# RESET CHAT
# ============================================================

def reset_chat():

    st.session_state["thread_id"] = generate_thread_id()

    st.session_state["message_history"] = []

    # Clear any pending HITL request
    st.session_state["pending_interrupt"] = None


# ============================================================
# LOAD PREVIOUS CONVERSATION
# ============================================================

def load_conversation(thread_id):

    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get("messages", [])


# ============================================================
# GET THREAD TITLE
# ============================================================

def get_thread_title(thread_id):

    messages = load_conversation(thread_id)

    for message in messages:

        if isinstance(message, HumanMessage):

            content = extract_text(message.content)

            if content:

                title = content.strip().replace("\n", " ")

                if len(title) > 35:
                    title = title[:35] + "..."

                return title

    return "New Conversation"


# ============================================================
# CHECK FOR HITL INTERRUPT
# ============================================================

def get_pending_interrupt(config):

    """
    Check whether the current LangGraph thread
    is paused because of an interrupt().
    """

    state = chatbot.get_state(config=config)

    # LangGraph stores interrupted tasks inside state.tasks
    for task in state.tasks:

        if task.interrupts:

            interrupt_obj = task.interrupts[0]

            return interrupt_obj.value

    return None


# ============================================================
# APP TITLE
# ============================================================

st.title("Agentic Chatbot With LangGraph")


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if "thread_id" not in st.session_state:

    st.session_state["thread_id"] = generate_thread_id()


if "message_history" not in st.session_state:

    st.session_state["message_history"] = []


if "chat_threads" not in st.session_state:

    st.session_state["chat_threads"] = get_all_threads()


# HITL state

if "pending_interrupt" not in st.session_state:

    st.session_state["pending_interrupt"] = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # PDF UPLOAD
    # --------------------------------------------------------

    st.subheader("📄 Upload PDF")

    uploaded_file = st.file_uploader(
        "Upload a PDF (Global Knowledge Base)",
        type=["pdf"],
        help="Upload a PDF and then ask questions about its content."
    )

    if uploaded_file is not None:

        temp_pdf_path = "uploaded_document.pdf"

        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Prevent processing the same file repeatedly
        if (
            "uploaded_pdf_name" not in st.session_state
            or
            st.session_state["uploaded_pdf_name"]
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


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    st.title("History")


    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button("New Chat"):

        reset_chat()

        st.rerun()


    # --------------------------------------------------------
    # PREVIOUS CONVERSATIONS
    # --------------------------------------------------------

    for thread_id in st.session_state["chat_threads"][::-1]:

        thread_title = get_thread_title(thread_id)

        if st.button(
            thread_title,
            key=f"thread_{thread_id}"
        ):

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

                # Ignore empty AI messages / tool-call messages
                if not content.strip():
                    continue

                tmp_mssg.append(
                    {
                        "role": role,
                        "content": content
                    }
                )

            st.session_state["message_history"] = tmp_mssg

            # Check if this conversation is currently waiting
            # for a HITL decision
            config = {
                "configurable": {
                    "thread_id": thread_id
                }
            }

            st.session_state["pending_interrupt"] = (
                get_pending_interrupt(config)
            )

            st.rerun()


# ============================================================
# MAIN CHAT HISTORY
# ============================================================

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# HITL APPROVAL UI
# ============================================================

if st.session_state["pending_interrupt"]:

    st.warning(
        "⏸️ The agent is waiting for your approval."
    )

    st.info(
        st.session_state["pending_interrupt"]
    )

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # APPROVE
    # --------------------------------------------------------

    with col1:

        if st.button(
            "✅ Approve",
            use_container_width=True
        ):

            config = {
                "configurable": {
                    "thread_id": st.session_state["thread_id"]
                }
            }

            with st.chat_message("assistant"):

                with st.spinner("Processing approval..."):

                    def approval_response_generator():

                        try:

                            for message_chunk, metadata in chatbot.stream(
                                Command(resume="yes"),
                                config=config,
                                stream_mode="messages"
                            ):

                                # Only stream chatbot responses
                                if metadata.get(
                                    "langgraph_node"
                                ) != "chat_node":

                                    continue

                                content = extract_text(
                                    message_chunk.content
                                )

                                if content:
                                    yield content

                        except Exception as e:

                            yield (
                                f"⚠️ Something went wrong: {str(e)}"
                            )


                    assistant_message = st.write_stream(
                        approval_response_generator()
                    )


            # Add final assistant response
            if assistant_message:

                st.session_state["message_history"].append(
                    {
                        "role": "assistant",
                        "content": assistant_message
                    }
                )


            # Clear HITL state
            st.session_state["pending_interrupt"] = None

            st.rerun()


    # --------------------------------------------------------
    # REJECT
    # --------------------------------------------------------

    with col2:

        if st.button(
            "❌ Reject",
            use_container_width=True
        ):

            config = {
                "configurable": {
                    "thread_id": st.session_state["thread_id"]
                }
            }

            with st.chat_message("assistant"):

                with st.spinner("Processing rejection..."):

                    def rejection_response_generator():

                        try:

                            for message_chunk, metadata in chatbot.stream(
                                Command(resume="no"),
                                config=config,
                                stream_mode="messages"
                            ):

                                if metadata.get(
                                    "langgraph_node"
                                ) != "chat_node":

                                    continue

                                content = extract_text(
                                    message_chunk.content
                                )

                                if content:
                                    yield content

                        except Exception as e:

                            yield (
                                f"⚠️ Something went wrong: {str(e)}"
                            )


                    assistant_message = st.write_stream(
                        rejection_response_generator()
                    )


            # Add final assistant response
            if assistant_message:

                st.session_state["message_history"].append(
                    {
                        "role": "assistant",
                        "content": assistant_message
                    }
                )


            # Clear HITL state
            st.session_state["pending_interrupt"] = None

            st.rerun()


# ============================================================
# USER INPUT
# ============================================================

user_input = st.chat_input(
    "Type anything here...",
    disabled=(
        st.session_state["pending_interrupt"] is not None
    )
)


# ============================================================
# PROCESS USER INPUT
# ============================================================

if user_input:

    # --------------------------------------------------------
    # ADD THREAD TO HISTORY
    # --------------------------------------------------------

    add_thread(
        st.session_state["thread_id"]
    )


    # --------------------------------------------------------
    # ADD USER MESSAGE TO UI
    # --------------------------------------------------------

    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input
        }
    )


    with st.chat_message("user"):

        st.text(user_input)


    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    config = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }


    # --------------------------------------------------------
    # RUN AGENT
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            def response_generator():

                try:

                    for message_chunk, metadata in chatbot.stream(
                        {
                            "messages": [
                                HumanMessage(
                                    content=user_input
                                )
                            ]
                        },
                        config=config,
                        stream_mode="messages"
                    ):

                        # Only stream chatbot node output
                        if metadata.get(
                            "langgraph_node"
                        ) != "chat_node":

                            continue


                        content = extract_text(
                            message_chunk.content
                        )


                        if content:

                            yield content


                except Exception as e:

                    yield (
                        f"⚠️ Something went wrong: {str(e)}"
                    )


            assistant_message = st.write_stream(
                response_generator()
            )


    # --------------------------------------------------------
    # CHECK WHETHER GRAPH IS WAITING FOR HUMAN
    # --------------------------------------------------------

    pending_interrupt = get_pending_interrupt(
        config
    )


    if pending_interrupt:

        # Save the interrupt in Streamlit session state
        st.session_state[
            "pending_interrupt"
        ] = pending_interrupt

    else:

        # Normal completed response
        if assistant_message:

            st.session_state["message_history"].append(
                {
                    "role": "assistant",
                    "content": assistant_message
                }
            )


    # --------------------------------------------------------
    # RERUN UI
    # --------------------------------------------------------

    st.rerun()