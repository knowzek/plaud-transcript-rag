from flask import Flask, request, jsonify
import chromadb
import os
from typing import List
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

app = Flask(__name__)

# Setup ChromaDB client with OpenAI Embeddings
embedding_function = OpenAIEmbeddingFunction(api_key=os.environ.get("CHROMA_OPENAI_API_KEY"))
client = chromadb.Client(Settings(
    allow_reset=True,
    persist_directory="/tmp/chroma"
))

collection = client.get_or_create_collection("transcripts", embedding_function=embedding_function)

# Chunking function
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

    except MemoryError:
        return jsonify({"error": "Out of memory during query"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 App running on port {port}")
    app.run(host="0.0.0.0", port=port)
