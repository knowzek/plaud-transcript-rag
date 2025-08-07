from flask import Flask, request, jsonify
import os
import openai
import pinecone
from typing import List
import uuid
import tiktoken
import re

app = Flask(__name__)

from pinecone import Pinecone, ServerlessSpec

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENVIRONMENT")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX_NAME", "transcripts-esteban")

# Create Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to the index (already created)
index = pc.Index(PINECONE_INDEX)


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# === UTILS ===

def make_ascii_id(text):
    # Replace non-ASCII characters with safe fallback
    ascii_text = text.encode("ascii", "ignore").decode()
    ascii_text = re.sub(r"[^\w\s-]", "", ascii_text)  # Remove any punctuation
    ascii_text = ascii_text.replace(" ", "_")  # Replace spaces with underscores
    return ascii_text

def chunk_transcript_by_tokens(text: str, max_tokens: int = 800) -> List[str]:
    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    words = text.split()
    chunks = []
    current_chunk = []

    current_tokens = 0
    for word in words:
        token_len = len(enc.encode(word + " "))
        if current_tokens + token_len > max_tokens:
            chunk = " ".join(current_chunk).strip()
            if chunk:
                chunks.append(chunk)
            current_chunk = [word]
            current_tokens = token_len
        else:
            current_chunk.append(word)
            current_tokens += token_len

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    enc = tiktoken.encoding_for_model("text-embedding-3-small")
    trimmed = []
    for t in texts:
        tokens = enc.encode(t)
        if len(tokens) > 8000:
            tokens = tokens[:8000]
            t = enc.decode(tokens)
        trimmed.append(t)

    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=trimmed
    )
    return [e.embedding for e in response.data]


# === ENDPOINTS ===

@app.route("/ingest", methods=["POST"])
def ingest():
    data = request.get_json()
    transcript = data.get("transcript", "")
    title = data.get("title", "Untitled")
    user_id = data.get("user_id")
    
    if not transcript:
        return jsonify({"error": "Missing transcript"}), 400
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    # Cap size
    if len(transcript) > 28000:
        transcript = transcript[:28000]

    chunks = chunk_transcript_by_tokens(transcript)
    texts_to_embed = [chunk for chunk in chunks if len(chunk) > 10]
    if not texts_to_embed:
        return jsonify({"error": "Transcript too short to chunk"}), 400

    print(f"📄 Ingesting: {title} → {len(texts_to_embed)} chunks")

    embeddings = embed_texts(texts_to_embed)

    to_upsert = []
    for i, (chunk, vector) in enumerate(zip(texts_to_embed, embeddings)):
        vector_id = f"{make_ascii_id(title)}-{uuid.uuid4().hex[:8]}"
        to_upsert.append((
            vector_id,
            vector,
            {
                "text": chunk,
                "source": title,
                "user_id": user_id  # ✅ attach user_id
            }
        ))

    try:
        index.upsert(vectors=to_upsert)
        print(f"✅ Ingested {len(to_upsert)} chunks")
        return jsonify({"status": "ingested", "chunks": len(to_upsert)}), 200
    except Exception as e:
        print(f"❌ Pinecone error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    query_text = data.get("query", "").strip()
    user_id = data.get("user_id")
    top_k = int(data.get("top_k", 3))

    if not query_text:
        return jsonify({"error": "Missing query"}), 400
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    try:
        embedding = embed_texts([query_text])[0]

        response = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            filter={"user_id": user_id}
        )

        results = response.get("matches", [])
        chunks = [match["metadata"]["text"] for match in results]
        sources = [match["metadata"].get("source", "") for match in results]

        return jsonify({
            "chunks": chunks,
            "sources": sources,
            "query": query_text
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# === SERVER ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 App running on port {port}")
    app.run(host="0.0.0.0", port=port)
