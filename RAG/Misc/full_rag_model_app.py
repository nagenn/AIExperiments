
import os
import json
import faiss
import numpy as np
from pathlib import Path
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import tornado.ioloop
import tornado.web

# --- Load and chunk document ---
def load_and_chunk_file(file_path, chunk_size=300):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

chunks = load_and_chunk_file("freshbite_faq.txt")

# --- Generate embeddings and create FAISS index ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = embedding_model.encode(chunks)
index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
index.add(np.array(chunk_embeddings))

# --- QA Model ---
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# --- Tornado Web Handlers ---
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

            # Embed and retrieve top chunks
            query_vec = embedding_model.encode([question])
            D, I = index.search(np.array(query_vec), k=3)
            retrieved_chunks = [chunks[i] for i in I[0]]
            combined_context = "\n\n".join(retrieved_chunks)

            # Run QA
            result = qa_pipeline(question=question, context=combined_context)
            self.write(json.dumps({
                "question": question,
                "answer": result["answer"],
                "confidence": round(result["score"], 2)
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
    print("Full RAG QA API running at http://localhost:8686")
    tornado.ioloop.IOLoop.current().start()
