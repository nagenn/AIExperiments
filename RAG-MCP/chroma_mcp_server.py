"""
MCP SERVER - chroma_mcp_server.py
Runs as a PERSISTENT SSE server on http://localhost:8001
Any MCP client can connect to it — visible in its own terminal window.

Run this first in its own terminal:
    python chroma_mcp_server.py

Requires:
    pip install mcp chromadb openai uvicorn
    export OPENAI_API_KEY=your_key_here
"""

import os
import chromadb
from openai import OpenAI
from mcp.server.fastmcp import FastMCP

# --- Configuration ---
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "freshbite_faq"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 3
PORT = 8001

# --- Initialise clients ---
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
try:
    collection = chroma_client.get_collection(COLLECTION_NAME)
    print(f"Connected to ChromaDB collection '{COLLECTION_NAME}' ({collection.count()} chunks)")
except Exception:
    raise RuntimeError(
        f"ChromaDB collection '{COLLECTION_NAME}' not found. "
        "Please run chunk_and_store.py first."
    )

# --- Create the MCP server ---
mcp = FastMCP("FreshBite FAQ Retriever", port=PORT)


@mcp.tool()
def search_faq(question: str) -> list[str]:
    """
    Search the FreshBite FAQ for chunks relevant to the given question.
    Returns a list of the most relevant text chunks.
    """
    print(f"\n[MCP Server] Received request: '{question}'")

    # Embed the question
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[question]
    )
    question_embedding = response.data[0].embedding

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=TOP_K
    )

    chunks = results["documents"][0]
    print(f"[MCP Server] Returning {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  {i}. {chunk}")

    return chunks


if __name__ == "__main__":
    print(f"Starting FreshBite MCP server on http://localhost:{PORT}/sse ...")
    mcp.run(transport="sse")
