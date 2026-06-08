import os
import sys
import uuid
from flask import Flask, request, jsonify, render_template, session
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from Retrieval_With_History_KW import ask_gpt_completions, augmented_query

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

DEFAULT_SYSTEM_PROMPT = (
    "You are a cooking assistant AI chat bot. A system trained for the sole purpose "
    "of providing cooking recipes based on user questions or requests. You can only "
    "solely answer and provide recipes using the VectorDB that was fed to your "
    "architecture during data ingestion. If you don't know the answer, you will say "
    "'I can only provide answers regarding cooking recipes.'"
)

# { chat_id: { "title": str, "history": [{"role": str, "content": str}] } }
chat_store: dict[str, dict] = {}


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

    full_history = chat_store[chat_id]["history"]

    # Slice history based on memory selector before passing to augmented_query
    if memory_turns == 0:
        chat_history = []
    elif memory_turns == -1:
        chat_history = full_history
    else:
        chat_history = full_history[-(memory_turns * 2):]

    try:
        # augmented_query now accepts chat_history directly and formats it properly
        user_prompt = augmented_query(query, chat_history)
        response_text, _ = ask_gpt_completions(system_prompt, user_prompt, model="gpt-4.1")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Store as clean role/content pairs for history (original query, not augmented)
    full_history.append({"role": "user", "content": query})
    full_history.append({"role": "assistant", "content": response_text})

    # Auto-title from first user message
    if chat_store[chat_id]["title"] == "New Chat":
        chat_store[chat_id]["title"] = query[:40] + ("…" if len(query) > 40 else "")

    return jsonify({"response": response_text, "title": chat_store[chat_id]["title"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5001)
