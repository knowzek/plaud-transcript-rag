from flask import Flask, request, jsonify
import chromadb
import os
from typing import List
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

app = Flask(__name__)

# Initialize Chroma

embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

client = chromadb.Client(Settings(allow_reset=True))
collection = client.get_or_create_collection("transcripts", embedding_function=embedding_function)

collection = client.get_or_create_collection("transcripts")

# Chunker
def chunk_transcript(text: str, max_chars: int = 500) -> List[str]:
    lines = text.split('\n')
    chunks = []
    current_chunk = ""
    for line in lines:
        if len(current_chunk) + len(line) < max_chars:
            current_chunk += " " + line.strip()
        else:
            chunks.append(current_chunk.strip())
            current_chunk = line.strip()
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Webhook to receive from Zapier
@app.route("/ingest", methods=["POST"])
def ingest_transcript():
    data = request.get_json()
    transcript_text = data.get("transcript", "")
    title = data.get("title", "Untitled")

    if not transcript_text:
        return jsonify({"error": "Missing transcript"}), 400

    chunks = chunk_transcript(transcript_text)

    for i, chunk in enumerate(chunks):
        doc_id = f"{title}-{i}"
        collection.add(
            documents=[chunk],
            metadatas=[{"source": title}],
            ids=[doc_id]
        )

    return jsonify({"status": "ingested", "chunks": len(chunks)}), 200

@app.route("/query", methods=["POST"])
def query_transcripts():
    try:
        data = request.get_json(force=True)
        query_text = data.get("query", "").strip()
        top_k = int(data.get("top_k", 3))

        if not query_text:
            return jsonify({"error": "Missing query"}), 400

        results = collection.query(
            query_texts=[query_text],
            n_results=top_k
        )

        chunks = results.get("documents", [[]])[0]
        sources = results.get("metadatas", [[]])[0]

        return jsonify({
            "chunks": chunks,
            "sources": sources,
            "query": query_text
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
