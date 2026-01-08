from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import base64
import time

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

# Use any working Stable Diffusion model
MODEL_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json",
}

class GenerateRequest(BaseModel):
    prompt: str

@app.post("/generate-image")
def generate_image(data: GenerateRequest):
    payload = {
        "inputs": data.prompt,
        "options": {"wait_for_model": True}
    }

    for _ in range(5):
        try:
            response = requests.post(MODEL_URL, headers=HEADERS, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                # Hugging Face returns base64 image
                if isinstance(result, dict) and "generated_image" in result:
                    img_bytes = base64.b64decode(result["generated_image"])
                    return {"image_bytes": result["generated_image"]}
                # Some models return a list
                if isinstance(result, list) and len(result) > 0 and "generated_image" in result[0]:
                    return {"image_bytes": result[0]["generated_image"]}
                return {"error": "No image returned"}
            if response.status_code == 503:
                time.sleep(3)
                continue
        except requests.exceptions.RequestException:
            time.sleep(2)
            continue
    raise HTTPException(status_code=504, detail="Model did not respond in time")
