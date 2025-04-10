import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from .model import PersonaModel
from .preprocess import TextPreprocessor

def train_model_from_data(texts, labels):
    """Train a persona model from text data and labels"""
    # Preprocess texts
    preprocessor = TextPreprocessor()
    processed_texts = [preprocessor.preprocess(text) for text in texts]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, labels, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = PersonaModel()
    model.train(X_train, y_train)
    
    # Evaluate model
    if X_test and y_test:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        print("Model Evaluation:")
        print(report)
    
    return model
