import os
from dotenv import load_dotenv
from google import genai  # Note: top-level import

load_dotenv(dotenv_path="../.env")

def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    client = genai.Client()
    return client
