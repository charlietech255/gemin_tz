from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise RuntimeError("HF_API_TOKEN not set")

# Use the v1 Responses API for GPT‑OSS
RESPONSES_URL = "https://router.huggingface.co/v1/responses"

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

class GenerateRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate(data: GenerateRequest):
    # Build a simple chat-ish structure
    payload = {
        "model": "openai/gpt-oss-120b:fastest",
        "input": data.prompt,
        "text_format": {"max_output_tokens": 256}
    }

    response = requests.post(RESPONSES_URL, headers=HEADERS, json=payload)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    result = response.json()

    # The Responses API returns an array of output events
    # The text content is usually in "output_text"
    text_output = result.get("output_text") or ""

    return {"output": text_output}
