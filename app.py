import streamlit as st
from database.mongodb import MongoDB
from services.chat_service import ChatService
from services.persona_service import PersonaService
from utils.helpers import get_session_id, initialize_chat_history, add_message_to_history
import pandas as pd

# Page configuration
st.set_page_config(page_title="AI Chatbot with Persona Analysis", layout="wide")

# Initialize services
db = MongoDB()
chat_service = ChatService()
persona_service = PersonaService()

# Initialize session state
initialize_chat_history()
session_id = get_session_id()

# App title
st.title("AI Chatbot with Persona Analysis")

# Create the sidebar for chat history and persona analysis
with st.sidebar:
    st.header("Chat History")
    
    # Get user's chat history from MongoDB
    mongo_chat_history = db.get_user_chat_history(session_id)
    
    # Display chat history in sidebar
    if mongo_chat_history:
        for chat in mongo_chat_history:
            st.text_area(
                "You:", 
                value=chat["user_message"], 
                height=100, 
                disabled=True,
                key=f"sidebar_user_{chat['_id']}"
            )
            st.text_area(
                "Bot:", 
                value=chat["bot_message"], 
                height=100, 
                disabled=True,
                key=f"sidebar_bot_{chat['_id']}"
            )
            st.divider()
    else:
        st.write("No chat history yet. Start chatting!")
    
    # Persona analysis section
    st.header("Persona Analysis")
    
    # Analyze button
    if st.button("Analyze My Persona"):
        user_messages = db.get_all_user_messages(session_id)
        if user_messages:
            with st.spinner("Analyzing your communication style..."):
                persona = persona_service.analyze_user_persona(user_messages)
                st.session_state.persona = persona
        else:
            st.session_state.persona = "Please chat with the bot first so we can analyze your persona."
    
    # Display persona if available
    if "persona" in st.session_state:
        st.subheader("Your Persona Analysis:")
        st.markdown(st.session_state.persona)
    
    # Add training data collection (for administrators)
    with st.expander("Admin: Add Training Data"):
        st.info("This section is for administrators to add labeled personas for model training.")
        
        # Select persona category
        persona_options = ["Technical Professional", "Customer Support", "Student", 
                           "Business Executive", "Creative Professional", "Casual User"]
        custom_persona = st.text_input("Or enter custom persona label:")
        
        selected_persona = st.selectbox("Select persona category:", persona_options)
        if custom_persona:
            selected_persona = custom_persona
        
        # Button to save current conversation as training data
        if st.button("Save Conversation as Training Data"):
            user_messages = db.get_all_user_messages(session_id)
            if user_messages and len(user_messages) >= 3:
                db.save_persona_label(session_id, user_messages, selected_persona)
                st.success(f"Conversation saved as example of '{selected_persona}' persona")
            else:
                st.error("Not enough messages to use as training data")

# Display chat history in main area
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_message = st.chat_input("Type your message here...")

if user_message:
    # Add user message to chat
    add_message_to_history("user", user_message)
    with st.chat_message("user"):
        st.markdown(user_message)
    
    # Get chat history for context
    mongo_chat_history = db.get_user_chat_history(session_id)
    
    # Get bot response
    with st.spinner("Thinking..."):
        bot_response = chat_service.get_bot_response(user_message, mongo_chat_history)
    
    # Add bot response to chat
    add_message_to_history("assistant", bot_response)
    with st.chat_message("assistant"):
        st.markdown(bot_response)
    
    # Save the exchange to MongoDB
    db.save_message(session_id, user_message, bot_response)

# Add information about the persona model
st.markdown("---")
with st.expander("About the Persona Analysis System"):
    st.write("""
    This chatbot uses a combination of machine learning and natural language processing to analyze user conversation patterns and determine persona characteristics.
    
    The system can either use:
    - A custom-trained machine learning model (when available)
    - OpenAI's API as a fallback option
    
    Administrators can contribute to the training dataset by labeling conversations, which improves the accuracy of the persona predictions over time.
    """)