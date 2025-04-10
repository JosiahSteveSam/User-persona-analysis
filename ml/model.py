import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import os

class PersonaModel:
    def __init__(self):
        self.model_path = "ml/persona_model.joblib"
        self.pipeline = None
        self.persona_categories = []
        self.load_model()
    
    def load_model(self):
        """Load the trained model if it exists"""
        if os.path.exists(self.model_path):
            try:
                loaded_model = joblib.load(self.model_path)
                self.pipeline = loaded_model["pipeline"]
                self.persona_categories = loaded_model["categories"]
                print("Model loaded successfully")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        else:
            print("No trained model found")
            return False
            
    def create_pipeline(self):
        """Create a new model pipeline"""
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
    def train(self, texts, labels):
        """Train the model with text data and persona labels"""
        if not self.pipeline:
            self.create_pipeline()
        
        # Get unique persona categories
        self.persona_categories = list(set(labels))
        
        # Train the model
        self.pipeline.fit(texts, labels)
        
        # Save the model
        model_data = {
            "pipeline": self.pipeline,
            "categories": self.persona_categories
        }
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model_data, self.model_path)
        print(f"Model trained and saved with {len(self.persona_categories)} categories")
        
    def predict(self, text):
        """Predict persona from text"""
        if not self.pipeline:
            raise ValueError("Model not loaded or trained")
            
        # For single text input, wrap in list
        if isinstance(text, str):
            predictions = self.pipeline.predict([text])
            # Return the predicted persona category
            return predictions[0]
        else:
            # For multiple texts
            predictions = self.pipeline.predict(text)
            return predictions
    
    def predict_proba(self, text):
        """Get probability distribution over persona categories"""
        if not self.pipeline:
            raise ValueError("Model not loaded or trained")
            
        # For single text input, wrap in list
        if isinstance(text, str):
            proba = self.pipeline.predict_proba([text])[0]
        else:
            proba = self.pipeline.predict_proba(text)
            
        # Create a dictionary mapping categories to probabilities
        result = {cat: float(prob) for cat, prob in zip(self.persona_categories, proba[0])}
        return result
