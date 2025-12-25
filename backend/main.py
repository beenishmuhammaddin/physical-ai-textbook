# backend/main.py
# Version: 1.0.1 - Deployed Dec 25, 2025
import os
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
from rag_system import RAGSystem

# Load .env
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

app = FastAPI(
    title="Physical AI Textbook RAG API",
    description="RAG-based chatbot for Physical AI & Robotics textbook",
    version="1.0.1"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
rag_system = None

class ChatRequest(BaseModel):
    message: str

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system
    if API_KEY:
        print("Initializing RAG system...")
        rag_system = RAGSystem(API_KEY)
        rag_system.load_documents()
        rag_system.generate_embeddings()
        print("RAG system ready!")
    else:
        print("Warning: API_KEY not found. RAG system not initialized.")

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

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    RAG-based chat endpoint that answers questions about the Physical AI textbook.
    """
    if not rag_system:
        return JSONResponse(
            status_code=503,
            content={"error": "RAG system not initialized. Check API key configuration."}
        )

    try:
        response = rag_system.chat(request.message)
        return {
            "message": request.message,
            "answer": response['answer'],
            "sources": response['sources']
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing chat request: {str(e)}"}
        )
