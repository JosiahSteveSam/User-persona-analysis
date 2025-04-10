import numpy as np
import pandas as pd
from textblob import TextBlob

class FeatureExtractor:
    def __init__(self):
        pass
    
    def extract_basic_features(self, text):
        """Extract basic text features"""
        features = {}
        
        # Text length
        features['text_length'] = len(text)
        
        # Word count
        features['word_count'] = len(text.split())
        
        # Average word length
        words = text.split()
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        
        # Sentiment analysis
        blob = TextBlob(text)
        features['sentiment_polarity'] = blob.sentiment.polarity
        features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        
        return features
    
    def extract_stylometric_features(self, text):
        """Extract stylometric features"""
        features = {}
        
        # Capitalization
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Punctuation frequency
        import re
        punctuation = re.findall(r'[^\w\s]', text)
        features['punct_ratio'] = len(punctuation) / len(text) if text else 0
        
        # Question mark and exclamation frequency
        features['question_ratio'] = text.count('?') / len(text) if text else 0
        features['exclamation_ratio'] = text.count('!') / len(text) if text else 0
        
        return features
    
    def extract_all_features(self, text):
        """Extract all features from text"""
        features = {}
        features.update(self.extract_basic_features(text))
        features.update(self.extract_stylometric_features(text))
        return features