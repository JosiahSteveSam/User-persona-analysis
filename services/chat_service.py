import openai
import streamlit as st

class ChatService:
    def __init__(self):
        openai.api_key = st.secrets["openai_api_key"]
    
    def get_bot_response(self, user_message, chat_history):
        """Get a response from the OpenAI API based on user input and chat history"""
        messages = []
        
        # Convert chat history to the format expected by OpenAI
        for msg in chat_history:
            if "user_message" in msg:
                messages.append({"role": "user", "content": msg["user_message"]})
            if "bot_message" in msg:
                messages.append({"role": "assistant", "content": msg["bot_message"]})
        
        # Add the current user message
        messages.append({"role": "user", "content": user_message})
        
        # Add system message to guide the assistant's behavior
        messages.insert(0, {
            "role": "system", 
            "content": "You are a helpful assistant. Be concise and friendly in your responses."
        })
        
        # Call the OpenAI API
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500
        )
        
        return response.choices[0].message.content