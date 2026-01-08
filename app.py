from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, requests, re

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

ASSISTANT_NAME = "Charlie"
DEVELOPER_NAME = "Charlie Syllas"

IDENTITY_PATTERN = re.compile(
    r"(who (are|made|created|built|trained|innovated) you|where are you from|what are you)",
    re.IGNORECASE
)

class GenerateRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate(req: GenerateRequest):
    prompt = req.prompt.strip()

    # 🔒 Identity enforcement
    if IDENTITY_PATTERN.search(prompt):
        return {
            "output": (
                f"## 🤖 {ASSISTANT_NAME}\n"
                f"I am **Charlie**, a programmer assistant developed by **{DEVELOPER_NAME}**."
            )
        }

    system_prompt = f"""
You are {ASSISTANT_NAME}, a professional technology assistant.

Rules:
- Answer ALL technology-related questions (programming, networking, cloud, security, etc.)
- Respond ONLY in Markdown
- Use clean formatting
- Use code blocks with language tags
- Never mention OpenAI or Hugging Face
- If asked about your identity, creator, or origin:
  Say: "I am Charlie, a programmer assistant developed by Charlie Syllas."
"""

    payload = {
        "model": "openai/gpt-oss-120b:fastest",
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }

    res = requests.post(RESPONSES_URL, headers=HEADERS, json=payload, timeout=120)
    if res.status_code != 200:
        raise HTTPException(status_code=res.status_code, detail=res.text)

    data = res.json()

    for item in data.get("output", []):
        if item.get("type") == "message" and item.get("role") == "assistant":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    return {"output": block["text"]}

    return {"output": "No response generated."}
