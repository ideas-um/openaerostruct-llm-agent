import os
import google.generativeai as genai

def get_model():
    genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
    model = genai.GenerativeModel("gemini-2.0-flash")
    return model
