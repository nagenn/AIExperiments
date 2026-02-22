you will need python 3.10 or later
Pip install ChromaDB mcp openai tornado
export OPENAI_API_KEY="sk-..." (your OpenAI API Key)
python chunk_and_store.py - this will create the vector store
python model-app.py — now a pure MCP client. It knows nothing about ChromaDB or embeddings. It just calls search_faq via the MCP protocol and passes the result to GPT. It will spin up the MCP server automatically
On browser goto http://127.0.0.1:8686
Ensure all the files (py, txt and html) are in the same folder (keep it in the download folder to keep it simple)

Python cli_client.py "Do you ship to UK?" 
##to demo that it calls the same MCP server (write once - use as many times as needed)

