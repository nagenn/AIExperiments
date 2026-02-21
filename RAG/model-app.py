"""
STEP 2 - Run this after chunk_and_store.py has been run.
Loads the ChromaDB vector store, and for each incoming question:
  1. Embeds the question
  2. Queries ChromaDB for the most similar chunks (retrieval)
  3. Sends those chunks + the question to GPT (augmented generation)

Usage:
    python model-app.py

Requires:
    pip install openai tornado chromadb
    export OPENAI_API_KEY=your_key_here
"""

import json
import tornado.ioloop
import tornado.web
import chromadb
from openai import OpenAI

client = OpenAI()

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "freshbite_faq"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 3  # Number of most relevant chunks to retrieve


# ---------------------------------------------------------------------------
# Load ChromaDB collection at startup
# ---------------------------------------------------------------------------

def load_collection(db_path, collection_name):
    try:
        chroma_client = chromadb.PersistentClient(path=db_path)
        collection = chroma_client.get_collection(collection_name)
        print(f"Connected to ChromaDB collection '{collection_name}' ({collection.count()} chunks)")
        return collection
    except Exception:
        raise RuntimeError(
            f"ChromaDB collection '{collection_name}' not found at '{db_path}'. "
            "Please run chunk_and_store.py first."
        )

collection = load_collection(CHROMA_DB_PATH, COLLECTION_NAME)


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def embed_question(question):
    """Embed the user's question using the same model as the chunks."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[question]
    )
    return response.data[0].embedding


def retrieve_top_chunks(question_embedding, top_k):
    """Query ChromaDB for the most similar chunks."""
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    # results["documents"] is a list of lists (one per query)
    return results["documents"][0]


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

            # Step 1: Embed the question
            question_embedding = embed_question(question)

            # Step 2: Retrieve the most relevant chunks from ChromaDB
            top_chunks = retrieve_top_chunks(question_embedding, TOP_K)
            retrieved_context = "\n".join(f"- {chunk}" for chunk in top_chunks)

            print(f"\nQuestion: {question}")
            print(f"Retrieved chunks:\n{retrieved_context}")

            # Step 3: Augment the prompt and generate an answer
            response = client.chat.completions.create(
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
                "retrieved_chunks": top_chunks  # useful to inspect during demo
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
