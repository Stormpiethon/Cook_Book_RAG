import os
import sys
import uuid
from flask import Flask, request, jsonify, render_template, session
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from Retrieval_KW import ask_gpt_completions, augmented_query

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

DEFAULT_SYSTEM_PROMPT = (
    "You are a cooking assistant AI chat bot. A system trained for the sole purpose "
    "of providing cooking recipes based on user questions or requests. You can only "
    "solely answer and provide recipes using the VectorDB that was fed to your "
    "architecture during data ingestion. If you don't know the answer, you will say "
    "'I can only provide answers regarding cooking recipes.'"
)

# { chat_id: { "title": str, "history": [{"role", "content", "display"}] } }
chat_store: dict[str, dict] = {}


def get_user_chats(sid: str) -> list[str]:
    """Return list of chat IDs belonging to this session (stored in session)."""
    return session.get("chats", [])


def ask_RAG_with_history(query: str, system_prompt: str, history: list, memory_turns: int) -> str:
    augmented = augmented_query(query)

    # Slice history to respect memory window (each turn = 1 user + 1 assistant msg)
    if memory_turns == 0:
        past_messages = []
    elif memory_turns == -1:
        past_messages = history  # all
    else:
        past_messages = history[-(memory_turns * 2):]

    past_context = "\n".join(
        f"{m['role'].upper()}: {m['display']}" for m in past_messages
    )
    user_prompt = (past_context + "\n\n" + augmented).strip() if past_context else augmented

    response_text, _ = ask_gpt_completions(system_prompt, user_prompt, model="gpt-4.1")
    return response_text, augmented


@app.route("/")
def index():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return render_template("index.html", default_system_prompt=DEFAULT_SYSTEM_PROMPT)


@app.route("/chats", methods=["GET"])
def list_chats():
    chat_ids = session.get("chats", [])
    result = []
    for cid in chat_ids:
        if cid in chat_store:
            result.append({"id": cid, "title": chat_store[cid]["title"]})
    return jsonify({"chats": result})


@app.route("/chats/new", methods=["POST"])
def new_chat():
    cid = str(uuid.uuid4())
    chat_store[cid] = {"title": "New Chat", "history": []}
    chats = session.get("chats", [])
    chats.append(cid)
    session["chats"] = chats
    session.modified = True
    return jsonify({"id": cid, "title": "New Chat"})


@app.route("/chats/<chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    chat_store.pop(chat_id, None)
    chats = session.get("chats", [])
    session["chats"] = [c for c in chats if c != chat_id]
    session.modified = True
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = (data.get("query") or "").strip()
    system_prompt = (data.get("system_prompt") or DEFAULT_SYSTEM_PROMPT).strip()
    chat_id = data.get("chat_id")
    memory_turns = int(data.get("memory_turns", 5))  # -1 = all, 0 = none

    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    if not chat_id or chat_id not in chat_store:
        return jsonify({"error": "Invalid chat ID."}), 400

    history = chat_store[chat_id]["history"]

    try:
        response_text, augmented = ask_RAG_with_history(
            query, system_prompt, history, memory_turns
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Store with display (original query) separate from augmented for context building
    history.append({"role": "user", "display": query, "content": augmented})
    history.append({"role": "assistant", "display": response_text, "content": response_text})

    # Auto-title from first user message
    if chat_store[chat_id]["title"] == "New Chat":
        chat_store[chat_id]["title"] = query[:40] + ("…" if len(query) > 40 else "")

    return jsonify({"response": response_text, "title": chat_store[chat_id]["title"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5001)
