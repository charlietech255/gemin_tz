# Charlie's AI Proxy API

Hii ni project rahisi sana ya FastAPI servers mbili zinazokusaidia kuunganisha apps zako na Large Language Models bila kuweka API keys zako hatarini upande wa frontend. Inafanya kazi kama daraja (proxy) kati ya user na AI providers.

Project hii imetengenezwa na Charlie (Charlie Syllas) kwa ajili ya kurahisisha workflow ya developers.

## Unachopata hapa

- **Version 1**: Inatumia OpenAI (gpt-4o-mini kwa default).
- **Version 2**: Inatumia Hugging Face router (model yoyote unayotaka, default ni gpt-oss-120b).
- **Security**: API keys zako zinabaki salama in server-side.
- **CORS Support**: ina support environment zote local na product level.
- **Identity Guard**: Version 2 ina uwezo wa kulinda utambulisho wa bot isitoe majibu ya hovyo kuhusu "nani kakuunda".

## Jinsi ya set (Setup Guide)

Fuata hatua hizi kusetup kwenye mashine yako au server:

```bash
# 1. Download project
git clone https://github.com/charlietech255/gemin_tz
cd gemin_tz

# 2. Install dependencies
pip install fastapi uvicorn requests python-dotenv pydantic

# 3. Tengeneza .env file
# ────────────────────────────────────────
# Kama unatumia OpenAI:
OPENAI_API_KEY=sk-...

# Kama unatumia Hugging Face:
HF_API_TOKEN=hf_...
# ────────────────────────────────────────

# 4. Washa server unayotaka
# Kwa OpenAI version
uvicorn openai_version:app --reload

# Au kwa Hugging Face version
uvicorn hf_version:app --reload
```

Server itakuwa hewani kwenye: [http://127.0.0.1:8000](http://127.0.0.1:8000)
Kama unataka kuchek documentation na kujaribu API: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Jinsi ya kuitumia (API Usage)

Unaweza kui call API yako kwa kutumia curl au fetch kwenye JavaScript:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Niambie faida tatu za kutumia Python"}'
```

Majibu yatakuja hivi:

```json
{
  "reply": "..."   // Kama unatumia OpenAI
  // au
  "output": "..."  // Kama unatumia HF version
}
```

## Tofauti ya hizi version mbili

| Feature | OpenAI Version | Hugging Face Router |
|---------|----------------|---------------------|
| Provider | OpenAI | Hugging Face |
| Model | gpt-4o-mini | openai/gpt-oss-120b |
| Identity Protection | Basic | Iko vizuri zaidi (Regex) |
| Field ya majibu | "reply" | "output" |
| System Prompt | Rahisi (Static) | Ina sheria nyingi zaidi |
| Timeout | Standard | 120 seconds |

## Tools zilizotumika

- FastAPI kwa ajili ya speed
- Requests kwa http request za AI models
- Pydantic kwa data validation
- Python-dotenv kwa .env setup

Iko fasta, haina gharama kubwa, na ni private. Tumia upendavyo!

— Charlie
