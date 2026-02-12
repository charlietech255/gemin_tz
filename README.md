```markdown
# Charlie's AI Proxy API

Two super-lightweight FastAPI servers that let your frontend / apps talk to big language models **without exposing API keys**.

Made by Charlie (Charlie Syllas) 🚀

## What you get

- Version 1 → talks to **OpenAI** (gpt-4o-mini by default)
- Version 2 → talks to **Hugging Face router** (using openai/gpt-oss-120b:fastest or whatever model you pick)
- Hides your API keys safely on the server
- CORS = wide open (good for local dev + deployed frontends)
- Very simple `/generate` endpoint
- Version 2 has built-in identity protection ("who are you" questions get clean answer)

## Quick start (both versions)

```bash
# 1. Get the code
git clone github.com/charlietech255/gemin_tz
cd main

# 2. Install
pip install fastapi uvicorn requests python-dotenv pydantic
# (or just: pip install -r requirements.txt if you have one)

# 3. Create .env file
# ────────────────────────────────────────
# For OpenAI version:
OPENAI_API_KEY=sk-...

# For Hugging Face router version:
HF_API_TOKEN=hf_...
# ────────────────────────────────────────

# 4. Run the one you want
# OpenAI version
uvicorn openai_version:app --reload    # (rename file to openai_version.py or whatever)

# OR Hugging Face version
uvicorn hf_version:app --reload        # (rename file to hf_version.py or similar)
```

Server usually starts at:  
http://127.0.0.1:8000

Docs + test page: http://127.0.0.1:8000/docs

## How to call it (both versions)

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain what a closure is in JavaScript in one sentence"}'
```

You get back something like:

```json
{
  "reply": "..."     // OpenAI version
  // or
  "output": "..."    // HF version
}
```

## Differences between the two versions

| Feature                     | OpenAI version              | Hugging Face router version          |
|-----------------------------|-----------------------------|---------------------------------------|
| Provider                    | OpenAI                      | Hugging Face (router)                 |
| Default model               | gpt-4o-mini                 | openai/gpt-oss-120b:fastest           |
| Identity protection         | basic                       | strong (regex + forced answer)        |
| Response field              | `"reply"`                   | `"output"`                            |
| System prompt control       | hard-coded simple           | more rules + Markdown enforcement     |
| Timeout                     | default requests            | 120 seconds                           |

## Made with

- FastAPI
- requests
- pydantic
- python-dotenv
- (very small amount of regex magic)

Quick, cheap, private.  
Enjoy!  
— Charlie
```

Feel free to split them into two separate repos / readmes later if you want — for now this keeps everything in one friendly place.

Let me know if you want it shorter, more serious, or with deployment instructions (Railway / Render / Fly.io etc.) 😄
