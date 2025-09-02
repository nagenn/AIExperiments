
import os
import faiss
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import uvicorn

# Load models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
language_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

# Load documents and build vector index
def load_documents(folder="docs"):
    docs = []
    for file in Path(folder).glob("*.txt"):
        text = file.read_text(encoding="utf-8")
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        docs.extend(chunks)
    return docs

documents = load_documents()
document_embeddings = embedding_model.encode(documents)
index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(np.array(document_embeddings))

# Retrieve top K chunks
def get_top_chunks(query, k=3):
    query_vec = embedding_model.encode([query])
    D, I = index.search(np.array(query_vec), k)
    return [documents[i] for i in I[0]]

# Construct augmented prompt
def build_prompt(context_chunks, question):
    context = "\n\n".join(context_chunks)
    return f"Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

# Generate response
def generate_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = language_model.generate(input_ids, max_new_tokens=300)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# FastAPI App
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def form():
    return '''
        <form action="/ask" method="post">
            <input type="text" name="query" style="width:400px"/>
            <input type="submit"/>
        </form>
    '''

@app.post("/ask", response_class=HTMLResponse)
async def ask(request: Request):
    form = await request.form()
    user_query = form["query"]
    top_chunks = get_top_chunks(user_query)
    full_prompt = build_prompt(top_chunks, user_query)
    response = generate_response(full_prompt)
    return f"<h3>Question:</h3><p>{user_query}</p><h3>Answer:</h3><p>{response}</p>"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
