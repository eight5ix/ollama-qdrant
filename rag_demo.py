import os
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List
import glob, pathlib
paths = glob.glob("data/**/*.txt", recursive=True)
DOCS = [{"id": i+1, "text": pathlib.Path(p).read_text(encoding='utf-8')} for i,p in enumerate(paths)]

# ---- Config ----
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")  # set if you enabled it
COLLECTION = os.environ.get("QDRANT_COLLECTION", "docs")

DOCS = [
    {"id": 1, "text": "Ollama runs local LLMs and provides an embeddings API."},
    {"id": 2, "text": "Qdrant is a vector database for similarity search."},
    {"id": 3, "text": "You can build a RAG pipeline by storing embeddings in Qdrant."},
]

def embed(texts: List[str]) -> List[List[float]]:
    vectors = []
    for t in texts:
        r = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": t},
            timeout=60,
        )
        r.raise_for_status()
        vectors.append(r.json()["embedding"])
    return vectors

def ensure_collection(client: QdrantClient, dim: int):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

def upsert_docs(client: QdrantClient, ids: List[int], vectors: List[List[float]], payloads: List[dict]):
    points = [PointStruct(id=i, vector=v, payload=p) for i, v, p in zip(ids, vectors, payloads)]
    client.upsert(collection_name=COLLECTION, points=points)

def search(client: QdrantClient, query: str, top_k: int = 3):
    qvec = embed([query])[0]
    return client.search(collection_name=COLLECTION, query_vector=qvec, limit=top_k)

def chat_with_context(question: str, contexts: List[str]) -> str:
    # Simple context stuffing -> Ollama chat
    system = "You are a helpful assistant. Use the provided context to answer."
    ctx_blob = "\n\n".join(f"- {c}" for c in contexts)
    prompt = f"Context:\n{ctx_blob}\n\nQuestion: {question}\nAnswer using only the context when possible."
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": "llama3.1", "messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": prompt}]},
        timeout=120,
    )
    r.raise_for_status()
    # Streaming may return multiple chunks; if so, concatenate
    data = r.json()
    if isinstance(data, dict) and "message" in data:
        return data["message"]["content"]
    if isinstance(data, list):
        return "".join(chunk.get("message", {}).get("content", "") for chunk in data)
    return str(data)

def main():
    # Connect to Qdrant
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Determine embed dimension dynamically
    dim = len(embed(["dimension probe"])[0])
    ensure_collection(client, dim)

    # Ingest example docs
    vectors = embed([d["text"] for d in DOCS])
    upsert_docs(client, [d["id"] for d in DOCS], vectors, [{"text": d["text"]} for d in DOCS])

    # Query
    query = "How do I build a RAG pipeline locally?"
    results = search(client, query, top_k=3)
    ctx = [hit.payload["text"] for hit in results]
    print("Nearest docs:")
    for hit in results:
        print(f"- score={hit.score:.4f}  text={hit.payload['text']}")
    print("\n---\nAnswer:")
    try:
        answer = chat_with_context(query, ctx)
        print(answer)
    except Exception as e:
        print(f"(Skipping chat; error: {e})")

if __name__ == "__main__":
    main()
