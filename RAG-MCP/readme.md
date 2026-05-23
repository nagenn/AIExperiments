FreshBite RAG + MCP Demo
Overview
This demo shows how a customer support AI assistant can be built using two modern AI engineering concepts:
	•	RAG (Retrieval Augmented Generation) — giving an AI model access to real business knowledge before generating an answer
	•	MCP (Model Context Protocol) — a standard protocol that decouples the retrieval layer from the application layer, so it can be reused by any client
The demo uses FreshBite Organics — a fictional organic food company — as the scenario.

Architecture
chunk_and_store.py          (run once)
        |
        v
   chroma_db/               (vector database)
        |
        v
chroma_mcp_server.py        (MCP server — persistent, port 8001)
        |
   MCP Protocol (SSE)
        |
   _____|_____
  |           |
model-app.py  cli_client.py
(web app)     (CLI tool)
  |
index.html
(browser UI)

Files
File
Purpose
freshbite_faq.txt
Source knowledge — FreshBite's product and shipping information
chunk_and_store.py
Chunks the FAQ, embeds each chunk via OpenAI, stores in ChromaDB
chroma_mcp_server.py
MCP server wrapping ChromaDB — exposes a search_faq tool
model-app.py
Tornado web server — MCP client that serves the browser UI
cli_client.py
Standalone CLI — second independent MCP client
index.html
Browser front end

python3.11 -m venv .venv311 (if the venv311 has not been created)
source .venv311/bin/activate (MCP server runs only on 3.10 or later)
Pip install ChromaDB
Pip install mcp
Pip install open
Pip install tornado
export OPENAI_API_KEY="sk-..."
Cd "MCP Demo"
python chunk_and_store.py
python model-app.py — now a pure MCP client. It knows nothing about ChromaDB or embeddings. It just calls search_faq via the MCP protocol and passes the result to GPT. It will spin up the MCP server automatically
On browser goto http://127.0.0.1:8686
Ensure all the files (py, txt and html) are in the same folder (keep it in the download folder to keep it simple)

Python cli_client.py "Do you ship to UK?" 
##to demo that it calls the same MCP server (write once - use as many times as needed)

deactivate

Prerequisites
	•	Python 3.11+
	•	A virtual environment (recommended)
	•	An OpenAI API key
Install dependencies
python -m venv .venv311
source .venv311/bin/activate       # Mac/Linux
.venv311\Scripts\activate          # Windows

pip install openai chromadb tornado mcp uvicorn

Running the Demo
Step 1 — Chunk and store (run once)
This reads freshbite_faq.txt, splits it into chunks, embeds them using OpenAI, and stores them in ChromaDB. Only needs to be re-run if the FAQ changes.
export OPENAI_API_KEY=your-key-here
python chunk_and_store.py
You should see:
Loaded 6 lines → 6 chunks
Embedding 6 chunks using 'text-embedding-3-small'...
Stored 6 chunks in ChromaDB at './chroma_db'
Done! Now run chroma_mcp_server.py to start the MCP server.

Step 2 — Start the MCP server (Terminal 1)
This runs persistently. Leave it running throughout the demo.
export OPENAI_API_KEY=your-key-here
python chroma_mcp_server.py
You should see:
Connected to ChromaDB collection 'freshbite_faq' (6 chunks)
Starting FreshBite MCP server on http://localhost:8001/sse ...
When a question arrives, this terminal will show exactly which chunks were retrieved — useful to show the class.

Step 3 — Start the web app (Terminal 2)
export OPENAI_API_KEY=your-key-here
python model-app.py
Open http://localhost:8686 in a browser and ask a question, e.g.:
Do your products contain nuts?

Step 4 — Run the CLI client (Terminal 3)
Demonstrates that a completely separate program can use the same MCP server.
export OPENAI_API_KEY=your-key-here
python cli_client.py "Do you ship to India?"


Key Concepts Illustrated
Chunking — splitting source documents into smaller pieces so relevant sections can be found individually
Embedding — converting text into vectors (lists of numbers) that capture semantic meaning, enabling similarity search
RAG — retrieving relevant chunks at query time and including them in the prompt, so the model answers from real business data rather than general training
MCP — a standard protocol so the retrieval server can be written once and used by any compliant client, with no custom integration code needed per client

Troubleshooting
"ChromaDB collection not found" — run chunk_and_store.py first
"All connection attempts failed" in cli_client.py — chroma_mcp_server.py is not running; start it in Terminal 1 first
"Missing API key" error — run export OPENAI_API_KEY=your-key-here in each terminal before running any script; the key must be set in every terminal session separately
you will need python 3.10 or later
Pip install ChromaDB mcp openai tornado
export OPENAI_API_KEY="sk-..." (your OpenAI API Key)
python chunk_and_store.py - this will create the vector store
python model-app.py — now a pure MCP client. It knows nothing about ChromaDB or embeddings. It just calls search_faq via the MCP protocol and passes the result to GPT. It will spin up the MCP server automatically
On browser goto http://127.0.0.1:8686
Ensure all the files (py, txt and html) are in the same folder (keep it in the download folder to keep it simple)

Python cli_client.py "Do you ship to UK?" 
##to demo that it calls the same MCP server (write once - use as many times as needed)

