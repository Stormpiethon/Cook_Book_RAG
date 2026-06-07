# Recipe Chatbot

A RAG-powered cooking assistant built with Flask, OpenAI, and Pinecone.

---

## Prerequisites

You need two API keys:

| Key | Where to get it |
|-----|----------------|
| `OPENAI_API_KEY` | platform.openai.com |
| `PINECONE_API_KEY` | app.pinecone.io |

---

## Option 1 — Docker (recommended)

### 1. Build the image
```bash
docker build -t recipe-chatbot .
```

### 2. Run
```bash
docker run -d -p 5001:5001 \
  -e OPENAI_API_KEY=sk-proj-... \
  -e PINECONE_API_KEY=pcsk_... \
  recipe-chatbot
```

### 3. Open
```
http://localhost:5001
```

### Stop
```bash
docker stop $(docker ps -q --filter ancestor=recipe-chatbot)
```

---

## Option 2 — Run locally (no Docker)

Requires **Python 3.9+**.

### 1. Set your API keys

Copy the example file and fill in your keys:
```bash
cp .env.example .env
# then edit .env with your keys
```

Or export them directly in your terminal:
```bash
export OPENAI_API_KEY=sk-proj-...
export PINECONE_API_KEY=pcsk_...
```

### 2. Run

**Mac / Linux:**
```bash
chmod +x run_local.sh
./run_local.sh
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\pip install -r requirements.txt
$env:OPENAI_API_KEY="sk-proj-..."
$env:PINECONE_API_KEY="pcsk_..."
venv\Scripts\python app.py
```

### 3. Open
```
http://localhost:5001
```

---

## Features

- **Chat sidebar** — previous conversations listed on the left; click to switch
- **New Chat** — start a fresh conversation any time
- **Memory selector** — control how many past turns the model remembers (None / Last 2 / Last 5 / Last 10 / Full history)
- **System Prompt** — collapsible panel to edit the assistant's behaviour; changes take effect on the next message

---

## Project structure

```
chatbot/
├── app.py              # Flask backend (routes, session management)
├── Retrieval_KW.py     # RAG logic (embeddings, Pinecone, GPT)
├── templates/
│   └── index.html      # Chat UI
├── static/
│   └── style.css       # Styling
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container build
├── .env.example        # API key template
└── run_local.sh        # One-command local runner (Mac/Linux)
```
