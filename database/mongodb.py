import pymongo
from datetime import datetime
import streamlit as st

class MongoDB:
    def __init__(self):
        self.client = pymongo.MongoClient(st.secrets["mongo_uri"])
        self.db = self.client[st.secrets["mongo_db_name"]]
        self.chat_collection = self.db[st.secrets["mongo_collection"]]
        self.persona_collection = self.db[st.secrets["persona_collection"]]
    
    def save_message(self, session_id, user_message, bot_message):
        """Save a message exchange to MongoDB"""
        document = {
            "session_id": session_id,
            "timestamp": datetime.now(),
            "user_message": user_message,
            "bot_message": bot_message
        }
        self.chat_collection.insert_one(document)
    
    def get_user_chat_history(self, session_id):
        """Get all chat history for a specific user session"""
        cursor = self.chat_collection.find({"session_id": session_id}).sort("timestamp", 1)
        return list(cursor)
    
    def get_all_user_messages(self, session_id):
        """Get all user messages for persona analysis"""
        cursor = self.chat_collection.find({"session_id": session_id})
        return [doc["user_message"] for doc in cursor]
    
    def save_persona_label(self, session_id, messages, persona_label):
        """Save a labeled persona for training data"""
        document = {
            "session_id": session_id,
            "timestamp": datetime.now(),
            "messages": messages,
            "combined_text": "\n".join(messages),
            "persona_label": persona_label
        }
        self.persona_collection.insert_one(document)
    
    def get_all_labeled_personas(self):
        """Get all labeled personas for model training"""
        cursor = self.persona_collection.find()
        return list(cursor)
