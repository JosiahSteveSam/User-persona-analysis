import pandas as pd
import argparse
from ml.train import train_model_from_data
from database.mongodb import MongoDB
import streamlit as st
import os
import json
from dotenv import load_dotenv

def load_config():
    """Load configuration from .env file or environment variables"""
    load_dotenv()
    
    # Create a dictionary to simulate streamlit secrets
    secrets = {
        "mongo_uri": os.getenv("MONGO_URI"),
        "mongo_db_name": os.getenv("MONGO_DB_NAME"),
        "mongo_collection": os.getenv("MONGO_COLLECTION"),
        "persona_collection": os.getenv("PERSONA_COLLECTION"),
    }
    
    # Create a class to simulate streamlit secrets
    class Secrets:
        def __getitem__(self, key):
            return secrets[key]
    
    # Set st.secrets to the simulated secrets
    st.secrets = Secrets()

def train_from_mongodb():
    """Train model from labeled data in MongoDB"""
    # Load configuration
    load_config()
    
    # Connect to MongoDB
    db = MongoDB()
    
    # Get labeled data
    labeled_data = db.get_all_labeled_personas()
    
    if not labeled_data:
        print("No labeled data found in MongoDB")
        return False
    
    # Extract texts and labels
    texts = [doc["combined_text"] for doc in labeled_data]
    labels = [doc["persona_label"] for doc in labeled_data]
    
    print(f"Training model with {len(texts)} labeled examples")
    
    # Train model
    train_model_from_data(texts, labels)
    return True

def train_from_csv(csv_path):
    """Train model from CSV file with texts and labels"""
    # Load CSV file
    try:
        df = pd.read_csv(csv_path)
        if "text" not in df.columns or "persona" not in df.columns:
            print("CSV must contain 'text' and 'persona' columns")
            return False
        
        # Extract texts and labels
        texts = df["text"].tolist()
        labels = df["persona"].tolist()
        
        print(f"Training model with {len(texts)} labeled examples from CSV")
        
        # Train model
        train_model_from_data(texts, labels)
        return True
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Train persona model")
    parser.add_argument("--csv", type=str, help="Path to CSV file with labeled data")
    
    args = parser.parse_args()
    
    if args.csv:
        success = train_from_csv(args.csv)
    else:
        success = train_from_mongodb()
    
    if success:
        print("Model training completed successfully")
    else:
        print("Model training failed")

if __name__ == "__main__":
    main()