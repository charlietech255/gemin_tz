from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests

# -----------------------------
# App setup
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Hugging Face config
# -----------------------------
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

if not HF_API_TOKEN:
    raise RuntimeError("HF_API_TOKEN environment variable not set")

MODEL_URL = "https://api-inference.huggingface.co/models/google/functiongemma-270m-it"

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json",
}

# -----------------------------
# Request schema
# -----------------------------
class GenerateRequest(BaseModel):
    prompt: str

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/generate")
def generate_text(data: GenerateRequest):
    payload = {
        "inputs": data.prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.7
        }
    }

    response = requests.post(
        MODEL_URL,
        headers=HEADERS,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=response.text
        )

    result = response.json()

    # Hugging Face usually returns a list
    if isinstance(result, list) and "generated_text" in result[0]:
        return {
            "output": result[0]["generated_text"]
        }

    return result
