"""
MCP SERVER - chroma_mcp_server.py
Wraps ChromaDB as an MCP server, exposing a single tool: search_faq(question)
Any MCP-compliant client can use this — not just this demo.

Run this first (in a separate terminal):
    python chroma_mcp_server.py

Requires:
    pip install mcp chromadb openai
    export OPENAI_API_KEY=your_key_here
"""

import os
from mcp.server.fastmcp import FastMCP
import chromadb
from openai import OpenAI

# --- Configuration ---
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "freshbite_faq"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 3

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
mcp = FastMCP("FreshBite FAQ Retriever")


@mcp.tool()
def search_faq(question: str) -> list[str]:
    """
    Search the FreshBite FAQ for chunks relevant to the given question.
    Returns a list of the most relevant text chunks.
    """
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
    print(f"[MCP Server] Question: '{question}' → returned {len(chunks)} chunks")
    return chunks


if __name__ == "__main__":
    print("Starting FreshBite MCP server (stdio transport)...")
    mcp.run(transport="stdio")
