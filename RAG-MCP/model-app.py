"""
MCP CLIENT + WEB SERVER - model-app.py
Connects to the persistent chroma_mcp_server.py over SSE.
No subprocess spawning — the MCP server runs independently.

Run this in a second terminal (after chroma_mcp_server.py is running):
    python model-app.py

Requires:
    pip install mcp openai tornado
    export OPENAI_API_KEY=your_key_here
"""

import os
import json
import asyncio
import tornado.ioloop
import tornado.web
from openai import OpenAI
from mcp.client.sse import sse_client
from mcp import ClientSession

# --- Configuration ---
CHAT_MODEL = "gpt-4o-mini"
MCP_SERVER_URL = "http://localhost:8001/sse"

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


async def retrieve_chunks_via_mcp(question: str) -> list[str]:
    """
    Connect to the persistent MCP server over SSE and call search_faq.
    No subprocess — just an HTTP connection to the running server.
    """
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
            return chunks


# ---------------------------------------------------------------------------
# Tornado handlers
# ---------------------------------------------------------------------------

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")


class QnAHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Content-Type", "application/json")

    async def post(self):
        try:
            data = json.loads(self.request.body)
            question = data.get("question", "")
            if not question:
                self.set_status(400)
                self.write(json.dumps({"error": "Missing 'question' in request"}))
                return

            # Step 1: Retrieve relevant chunks via MCP SSE connection
            top_chunks = await retrieve_chunks_via_mcp(question)
            retrieved_context = "\n".join(f"- {chunk}" for chunk in top_chunks)

            print(f"\n[Web App] Question: {question}")
            print(f"[Web App] Retrieved via MCP:\n{retrieved_context}")

            # Step 2: Augment the prompt and generate an answer
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
                        "content": f"Context:\n{retrieved_context}\n\nQuestion: {question}"
                    }
                ],
                max_tokens=256,
                temperature=0.2
            )

            answer = response.choices[0].message.content.strip()
            self.write(json.dumps({
                "question": question,
                "answer": answer,
                "retrieved_chunks": top_chunks
            }))

        except Exception as e:
            self.set_status(500)
            self.write(json.dumps({"error": str(e)}))


def make_app():
    return tornado.web.Application([
        (r"/", IndexHandler),
        (r"/ask", QnAHandler),
    ], template_path=".")


if __name__ == "__main__":
    app = make_app()
    app.listen(8686)
    print("Tornado QA API running at http://localhost:8686")
    print(f"Connecting to MCP server at {MCP_SERVER_URL}")
    tornado.ioloop.IOLoop.current().start()
