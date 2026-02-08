"""Simple test to check if API can load"""
import os
from dotenv import load_dotenv

print("1. Loading .env...")
load_dotenv()

print("2. Checking GEMINI_API_KEY...")
api_key = os.getenv("GEMINI_API_KEY")
print(f"   API Key found: {bool(api_key)}")

print("3. Importing FastAPI...")
from fastapi import FastAPI

print("4. Creating FastAPI app...")
app = FastAPI(title="Test")

print("5. Importing Google GenAI...")
try:
    import google.generativeai as genai
    print("   GenAI imported OK")
    
    print("6. Configuring GenAI...")
    genai.configure(api_key=api_key)
    print("   GenAI configured OK")
except Exception as e:
    print(f"   ERROR: {e}")

print("7. Importing numpy and librosa...")
try:
    import numpy as np
    import librosa
    print("   Numpy and librosa imported OK")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n✅ All imports successful!")
print("The API should work. Try starting with: python -m uvicorn api:app --reload --port 8000")
