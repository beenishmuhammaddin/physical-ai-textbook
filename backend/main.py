# backend/main.py
import os
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import requests

# Load .env
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Backend Running Successfully!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/generate")
def generate(prompt: str = Query(..., description="The prompt to send to Gemini API")):
    """
    Real Gemini API call using Google Cloud API key.
    Returns fallback message if request fails.
    """
    if not API_KEY:
        return JSONResponse(
            status_code=200,
            content={"prompt": prompt, "response": "API key missing."}
        )

    try:
        # Correct Gemini endpoint with API key as query param
        url = f"https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generate?key={API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "prompt": prompt,
            "temperature": 0.7,
            "maxOutputTokens": 256
        }

        response = requests.post(url, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()

        # Extract generated text
        output_text = data.get("candidates", [{}])[0].get("content", "")
        return {"prompt": prompt, "response": output_text}

    except requests.exceptions.RequestException as e:
        # Fallback on request error
        return JSONResponse(
            status_code=200,
            content={"prompt": prompt, "response": f"API request failed: {str(e)}"}
        )
