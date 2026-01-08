from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import time

# -----------------------------
# App setup
# -----------------------------
app = FastAPI()

# Allow frontend JS requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Hugging Face config
# -----------------------------
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise RuntimeError("HF_API_TOKEN environment variable not set")

MODEL_URL = "https://router.huggingface.co/models/google/functiongemma-270m-it"

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
def root():
    return {"status": "running"}

@app.post("/generate")
def generate(data: GenerateRequest):

    # Instruction-style prompt for FunctionGemma
    prompt = f"""### Instruction:
{data.prompt}

### Response:
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.7
        },
        "options": {
            "wait_for_model": True
        }
    }

    # Retry logic (model cold start)
    for attempt in range(5):
        try:
            response = requests.post(
                MODEL_URL,
                headers=HEADERS,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()

                if isinstance(result, list) and "generated_text" in result[0]:
                    # Remove prompt from output
                    clean_output = result[0]["generated_text"].replace(prompt, "").strip()
                    return {"output": clean_output}

                return result

            # Model loading → retry
            if response.status_code == 503:
                time.sleep(3)
                continue

            raise HTTPException(
                status_code=response.status_code,
                detail=response.text
            )

        except requests.exceptions.RequestException as e:
            # Network / timeout errors → retry
            time.sleep(2)
            continue

    raise HTTPException(status_code=504, detail="Model did not respond in time")
