"""
STEP 1 - Run this first!
Reads freshbite_faq.txt, splits it into chunks, embeds each chunk
using OpenAI, and stores the results in a ChromaDB vector database.

Usage:
    python chunk_and_store.py

Requires:
    pip install openai chromadb
    export OPENAI_API_KEY=your_key_here
"""

import chromadb
from openai import OpenAI

client = OpenAI()

INPUT_FILE = "freshbite_faq.txt"
CHUNK_SIZE = 1          # Number of lines per chunk (adjust for larger documents)
EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_DB_PATH = "./chroma_db"      # ChromaDB will persist data here
COLLECTION_NAME = "freshbite_faq"


def load_and_chunk(file_path, chunk_size):
    """Read the file and split into chunks of chunk_size lines."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]  # skip blank lines

    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunk = " ".join(lines[i:i + chunk_size])
        chunks.append(chunk)

    print(f"Loaded {len(lines)} lines → {len(chunks)} chunks")
    return chunks


def embed_chunks(chunks):
    """Send all chunks to OpenAI embeddings API in one batch call."""
    print(f"Embedding {len(chunks)} chunks using '{EMBEDDING_MODEL}'...")
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=chunks
    )
    embeddings = [item.embedding for item in response.data]
    print("Embedding complete.")
    return embeddings


def store_in_chroma(chunks, embeddings):
    """Store chunks and their embeddings in ChromaDB."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Delete existing collection if it exists (so re-runs start fresh)
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass  # Collection didn't exist yet, that's fine

    collection = chroma_client.create_collection(COLLECTION_NAME)

    # ChromaDB expects ids, embeddings, and documents as separate lists
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB at '{CHROMA_DB_PATH}'")


if __name__ == "__main__":
    chunks = load_and_chunk(INPUT_FILE, CHUNK_SIZE)
    embeddings = embed_chunks(chunks)
    store_in_chroma(chunks, embeddings)
    print("\nDone! Now run model-app.py to start the QA server.")
