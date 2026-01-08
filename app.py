from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
import time

app = FastAPI()

# Allow frontend JS
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

    # Retry logic for cold starts
    for attempt in range(5):
        try:
            response = requests.post(RESPONSES_URL, headers=HEADERS, json=payload, timeout=120)
            if response.status_code == 200:
                result = response.json()
                # Hugging Face Responses API returns outputs as a list of dicts
                outputs = result.get("outputs", [])
                if outputs:
                    # Usually the first output contains "type":"generated_text"
                    for out in outputs:
                        if out.get("type") == "generated_text" and "text" in out:
                            return {"output": out["text"]}
                # Fallback
                return {"output": str(result)}
            elif response.status_code == 503:
                # Model loading → wait and retry
                time.sleep(3)
                continue
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
        except requests.exceptions.RequestException:
            time.sleep(2)
            continue

    raise HTTPException(status_code=504, detail="Model did not respond in time")
