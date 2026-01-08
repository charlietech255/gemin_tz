from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
import re

app = FastAPI()

# CORS (allow browser requests)
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

BOT_OWNER = "Charlie Syllas"

ALLOWED_TOPICS = [
    "programming", "web", "javascript", "python",
    "html", "css", "technology", "software", "api"
]

IDENTITY_QUESTIONS = re.compile(
    r"(who (made|created|built|taught|trained) you|who is your creator)",
    re.IGNORECASE
)

class GenerateRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate(data: GenerateRequest):
    user_prompt = data.prompt.strip()

    # 🔐 Identity override
    if IDENTITY_QUESTIONS.search(user_prompt):
        return {"output": f"**I was created and taught by {BOT_OWNER}.**"}

    # 🔐 Topic restriction
    if not any(t in user_prompt.lower() for t in ALLOWED_TOPICS):
        return {
            "output": (
                "❌ **Topic not allowed**\n\n"
                "I only answer questions about:\n"
                "- Programming\n"
                "- Web development\n"
                "- Technology"
            )
        }

    system_prompt = f"""
You are a helpful assistant created by {BOT_OWNER}.

Rules:
- Respond ONLY in valid Markdown
- Use headings, lists, tables when helpful
- Keep formatting clean and readable
- Never mention OpenAI or Hugging Face
"""

    payload = {
        "model": "openai/gpt-oss-120b:fastest",
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    response = requests.post(
        RESPONSES_URL,
        headers=HEADERS,
        json=payload,
        timeout=120
    )

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    result = response.json()

    # ✅ Extract assistant output
    for item in result.get("output", []):
        if item.get("type") == "message" and item.get("role") == "assistant":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    return {"output": block["text"].strip()}

    return {"output": "No response generated."}
