from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests

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

RESPONSES_URL = "https://router.huggingface.co/v1/responses"

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

class GenerateRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate(data: GenerateRequest):
    payload = {
        "model": "openai/gpt-oss-120b:fastest",
        "input": data.prompt
    }

    response = requests.post(RESPONSES_URL, headers=HEADERS, json=payload, timeout=120)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    result = response.json()

    # ✅ Extract assistant message text
    try:
        for item in result.get("output", []):
            if item.get("type") == "message" and item.get("role") == "assistant":
                content = item.get("content", [])
                for block in content:
                    if block.get("type") == "output_text":
                        return {"output": block.get("text", "").strip()}
    except Exception:
        pass

    # Fallback (for debugging)
    return {"output": "No assistant message found", "raw": result}
