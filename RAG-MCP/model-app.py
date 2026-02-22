"""
MCP CLIENT + WEB SERVER - model-app.py
Acts as an MCP client: connects to chroma_mcp_server.py to retrieve
relevant chunks, then sends them to OpenAI GPT to generate an answer.

Run this after chroma_mcp_server.py is available:
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
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- Configuration ---
CHAT_MODEL = "gpt-4o-mini"
MCP_SERVER_SCRIPT = "chroma_mcp_server.py"  # path to your MCP server

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# MCP server connection parameters
server_params = StdioServerParameters(
    command="python",
    args=[MCP_SERVER_SCRIPT]
)


async def retrieve_chunks_via_mcp(question: str) -> list[str]:
    """
    Connect to the MCP server, call the search_faq tool, and return chunks.
    The model-app has no knowledge of ChromaDB or embeddings — it just speaks MCP.
    """
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Call the MCP tool — this is the only thing the client knows about
            result = await session.call_tool("search_faq", {"question": question})

            # Extract the text content from the MCP response
            chunks = []
            for content_item in result.content:
                if hasattr(content_item, "text"):
                    # The tool returns a JSON list of strings
                    import ast
                    try:
                        parsed = ast.literal_eval(content_item.text)
                        if isinstance(parsed, list):
                            chunks.extend(parsed)
                        else:
                            chunks.append(content_item.text)
                    except Exception:
                        chunks.append(content_item.text)

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

            # Step 1: Retrieve relevant chunks via MCP (no ChromaDB code here!)
            top_chunks = await retrieve_chunks_via_mcp(question)
            retrieved_context = "\n".join(f"- {chunk}" for chunk in top_chunks)

            print(f"\nQuestion: {question}")
            print(f"Retrieved via MCP:\n{retrieved_context}")

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
    tornado.ioloop.IOLoop.current().start()
