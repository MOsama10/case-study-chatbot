# list_models.py
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get your GEMINI_API_KEY from .env
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

genai.configure(api_key=api_key)

for m in genai.list_models():
    print(f"{m.name} | Supported methods: {m.supported_generation_methods}")
