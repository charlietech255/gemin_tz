from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import time

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise RuntimeError("HF_API_TOKEN environment variable not set")

# NEW router endpoint
MODEL_URL = "https://api-inference.huggingface.co/models/google/functiongemma-270m-it"

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json",
}

class GenerateRequest(BaseModel):
    prompt: str

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/generate")
def generate(data: GenerateRequest):
    # Instruction prompt
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
        "options": {"wait_for_model": True}
    }

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
                # The router endpoint sometimes returns nested objects
                if isinstance(result, dict) and "generated_text" in result:
                    return {"output": result["generated_text"].replace(prompt, "").strip()}

                if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                    return {"output": result[0]["generated_text"].replace(prompt, "").strip()}

                # Fallback: send full JSON if structure unknown
                return {"output": str(result)}

            if response.status_code == 503:
                # model is loading
                time.sleep(3)
                continue

            raise HTTPException(status_code=response.status_code, detail=response.text)

        except requests.exceptions.RequestException as e:
            time.sleep(2)
            continue

    raise HTTPException(status_code=504, detail="Model did not respond in time")
