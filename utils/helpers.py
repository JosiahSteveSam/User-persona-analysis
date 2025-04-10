import uuid
import streamlit as st

def get_session_id():
    """Get or create a unique session ID for the current user"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def initialize_chat_history():
    """Initialize chat history in session state if it doesn't exist"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

def add_message_to_history(role, content):
    """Add a message to the chat history in the session state"""
    st.session_state.messages.append({"role": role, "content": content})