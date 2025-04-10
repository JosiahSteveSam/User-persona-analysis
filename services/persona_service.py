import openai
import streamlit as st
from ml.model import PersonaModel
from ml.preprocess import TextPreprocessor

class PersonaService:
    def __init__(self):
        openai.api_key = st.secrets["openai_api_key"]
        self.model = PersonaModel()
        self.preprocessor = TextPreprocessor()
        self.use_custom_model = self.model.load_model()  # Check if model exists
    
    def analyze_user_persona(self, messages):
        """Analyze user messages to determine their persona"""
        if not messages or len(messages) < 3:
            return "Not enough data to analyze persona yet. Please continue chatting."
        
        # Combine all user messages for analysis
        user_text = "\n".join(messages)
        
        # If custom model is available and loaded, use it
        if self.use_custom_model:
            try:
                # Preprocess the text
                processed_text = self.preprocessor.preprocess(user_text)
                
                # Get prediction and probabilities
                predicted_persona = self.model.predict(processed_text)
                probabilities = self.model.predict_proba(processed_text)
                
                # Format the result
                result = f"### User Persona: {predicted_persona}\n\n"
                result += "#### Persona Confidence Scores:\n"
                
                # Sort probabilities by confidence
                sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                for persona, prob in sorted_probs:
                    result += f"- {persona}: {prob:.2%}\n"
                
                return result
            except Exception as e:
                # Fall back to OpenAI if there's an error with custom model
                print(f"Error using custom model: {e}")
                return self._analyze_with_openai(user_text)
        else:
            # Use OpenAI's API if no custom model is available
            return self._analyze_with_openai(user_text)
    
    def _analyze_with_openai(self, user_text):
        """Analyze user persona using OpenAI's API"""
        system_prompt = """
        Analyze the following chat messages from a user and create a detailed persona profile.
        Include:
        1. Demographics (likely age range, possible profession)
        2. Communication style (formal/informal, verbose/concise)
        3. Primary interests based on conversation topics
        4. Technical knowledge level
        5. Personality traits evident from the text
        
        Keep the analysis professional and evidence-based. Don't make assumptions not supported by the data.
        """
        
        response = openai.chat.completions.create(
            model="gpt-4",  # Using a more powerful model for analysis
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here are the user messages to analyze:\n{user_text}"}
            ],
            max_tokens=800,
            temperature=0.7
        )
        
        return response.choices[0].message.content
