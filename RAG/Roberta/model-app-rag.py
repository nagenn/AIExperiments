
import os
import json
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
import tornado.ioloop
import tornado.web

# Load and chunk the document
def load_and_chunk_file(file_path, chunk_size=500):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

chunks = load_and_chunk_file("freshbite_faq.txt")

# Create vector index using sentence embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = embedding_model.encode(chunks)
index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
index.add(np.array(chunk_embeddings))

# Load Roberta-based QA model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")

# Reranking function for better context relevance
def keyword_overlap_score(chunk, question):
    chunk_words = set(chunk.lower().split())
    question_words = set(question.lower().split())
    return len(chunk_words & question_words)

# Tornado Web Handlers
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

            # Retrieve top-k chunks
            query_vec = embedding_model.encode([question])
            D, I = index.search(np.array(query_vec), k=5)
            retrieved_chunks = [chunks[i] for i in I[0] if len(chunks[i].strip()) > 20]

            # Re-rank by keyword overlap
            retrieved_chunks.sort(key=lambda c: keyword_overlap_score(c, question), reverse=True)

            # Search best answer
            best_answer = ""
            best_score = 0
            for context in retrieved_chunks:
                result = qa_pipeline(question=question, context=context)
                if result["score"] > best_score:
                    best_answer = result["answer"]
                    best_score = result["score"]

            if best_score < 0.4:
                final_answer = "I'm not confident about the answer based on available information."
            else:
                final_answer = best_answer

            self.write(json.dumps({
                "question": question,
                "answer": final_answer
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
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)

    app = make_app()
    app.listen(8686)
    print("Improved Roberta RAG App running at http://localhost:8686")
    tornado.ioloop.IOLoop.current().start()
