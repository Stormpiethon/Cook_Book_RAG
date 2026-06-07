#!/bin/bash
set -e

# ── Recipe Chatbot — local runner (no Docker needed) ──────────────────────────

# 1. Check Python
if ! command -v python3 &>/dev/null; then
  echo "ERROR: python3 not found. Install Python 3.9+ and try again."
  exit 1
fi

PYTHON=$(command -v python3)
echo "Using Python: $($PYTHON --version)"

# 2. Load .env if present
if [ -f .env ]; then
  echo "Loading .env ..."
  export $(grep -v '^#' .env | xargs)
fi

# 3. Check required env vars
if [ -z "$OPENAI_API_KEY" ] || [ -z "$PINECONE_API_KEY" ]; then
  echo ""
  echo "ERROR: OPENAI_API_KEY and PINECONE_API_KEY must be set."
  echo "  Option A — copy .env.example to .env and fill in your keys, then re-run this script."
  echo "  Option B — export them manually:"
  echo "             export OPENAI_API_KEY=sk-proj-..."
  echo "             export PINECONE_API_KEY=pcsk_..."
  echo ""
  exit 1
fi

# 4. Create venv if it doesn't exist
if [ ! -d "venv" ]; then
  echo "Creating virtual environment ..."
  $PYTHON -m venv venv
fi

# 5. Install dependencies
echo "Installing dependencies ..."
venv/bin/pip install --quiet --upgrade pip
venv/bin/pip install --quiet -r requirements.txt

# 6. Run
echo ""
echo "Starting Recipe Chatbot at http://localhost:5001"
echo "Press Ctrl+C to stop."
echo ""
venv/bin/python app.py
