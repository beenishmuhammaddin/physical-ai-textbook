from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from openai import OpenAI

# Setup FastAPI
app = FastAPI(title="Physical AI Chatbot")

# Example Qdrant Client (replace with your credentials)
qdrant_client = QdrantClient(
    url="https://aca1129d-4d5e-4a41-8c11-7218f4c4302f.europe-west3-0.gcp.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.q6nMxpGbdTiRZnT7acBbqzxyeW-F3EBy9ezngBJBgms"
)

# OpenAI Client
openai_client = OpenAI(api_key="AIzaSyDznVRtphVgTiZufPJ4Geu-xF7SLOepzeo")

# Request model
class AskRequest(BaseModel):
    question: str
    selected_text: str

@app.post("/ask")
def ask(request: AskRequest):
    # TODO: Add vector search in Qdrant for selected_text
    # TODO: Generate AI answer using OpenAI / Agents SDK
    answer = f"Answering: {request.question} based on selected text."
    return {"answer": answer}
