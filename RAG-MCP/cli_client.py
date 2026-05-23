"""
STANDALONE MCP CLIENT - cli_client.py
Connects to the persistent chroma_mcp_server.py over SSE.
Demonstrates "write once, use anywhere" — a completely separate program
using the same MCP server as model-app.py.

Usage:
    python cli_client.py "Do you ship to India?"

Requires:
    pip install mcp openai
    export OPENAI_API_KEY=your_key_here
"""

import os
import sys
import asyncio
from openai import OpenAI
from mcp.client.sse import sse_client
from mcp import ClientSession

# --- Configuration ---
CHAT_MODEL = "gpt-4o-mini"
MCP_SERVER_URL = "http://localhost:8001/sse"

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


async def ask(question: str):
    print(f"\nQuestion: {question}")
    print("-" * 50)

    # Connect to the persistent MCP server over SSE
    async with sse_client(MCP_SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("search_faq", {"question": question})

            chunks = []
            for item in result.content:
                if hasattr(item, "text"):
                    import ast
                    try:
                        parsed = ast.literal_eval(item.text)
                        if isinstance(parsed, list):
                            chunks.extend(parsed)
                        else:
                            chunks.append(item.text)
                    except Exception:
                        chunks.append(item.text)

    print(f"Retrieved {len(chunks)} chunks from MCP server:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  {i}. {chunk}")
    print()

    # Generate answer using OpenAI
    context = "\n".join(f"- {chunk}" for chunk in chunks)
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant for FreshBite Organics. "
                    "Answer questions using ONLY the context provided. "
                    "If the answer is not in the context, say 'I don't have that information.'"
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        max_tokens=256,
        temperature=0.2
    )

    print(f"Answer: {response.choices[0].message.content.strip()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cli_client.py \"your question here\"")
        print("Example: python cli_client.py \"Do you ship to India?\"")
        sys.exit(1)

    question = sys.argv[1]
    asyncio.run(ask(question))
