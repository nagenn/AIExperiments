
# RAG Tornado App with FAISS retrieval, embeddings, and extractive QA reranking
import os
import json
import glob
import pickle
import time
import tornado.ioloop
import tornado.web
from typing import List, Dict, Any

from tqdm import tqdm
import numpy as np

# PDF
import fitz  # PyMuPDF

# Embeddings and retrieval
from sentence_transformers import SentenceTransformer
import faiss

# QA model
from transformers import pipeline

DOCS_DIR = "docs"
INDEX_DIR = "rag_index"
CHUNKS_PKL = os.path.join(INDEX_DIR, "chunks.pkl")
FAISS_INDEX = os.path.join(INDEX_DIR, "faiss.index")
META_PKL = os.path.join(INDEX_DIR, "meta.pkl")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize models lazily
embedding_model = None
qa_pipeline = None
faiss_index = None
chunks = []
metas = []

# ----------------------------- Utils -----------------------------

def ensure_dirs():
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR, exist_ok=True)
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR, exist_ok=True)


def load_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    texts = []
    for page in doc:
        text = page.get_text()
        if text:
            texts.append(text)
    doc.close()
    return "
".join(texts)


def load_txt_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        with open(path, "r", errors="ignore") as f:
            return f.read()


def split_into_chunks(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    text = text.replace("
", "
").replace("
", "
")
    paragraphs = [p.strip() for p in text.split("

") if p.strip()]
    chunks_local = []
    for p in paragraphs:
        if len(p) <= chunk_size:
            chunks_local.append(p)
        else:
            start = 0
            while start < len(p):
                end = start + chunk_size
                chunks_local.append(p[start:end])
                start = max(end - overlap, start + 1)
    return chunks_local


def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
    return embedding_model


def get_qa_pipeline():
    global qa_pipeline
    if qa_pipeline is None:
        qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return qa_pipeline


def build_or_load_index(force_rebuild: bool = False) -> None:
    global faiss_index, chunks, metas
    ensure_dirs()

    if (not force_rebuild) and os.path.exists(FAISS_INDEX) and os.path.exists(CHUNKS_PKL) and os.path.exists(META_PKL):
        faiss_index = faiss.read_index(FAISS_INDEX)
        with open(CHUNKS_PKL, "rb") as f:
            chunks = pickle.load(f)
        with open(META_PKL, "rb") as f:
            metas = pickle.load(f)
        return

    # Gather documents
    files = []
    files.extend(glob.glob(os.path.join(DOCS_DIR, "**", "*.txt"), recursive=True))
    files.extend(glob.glob(os.path.join(DOCS_DIR, "**", "*.md"), recursive=True))
    files.extend(glob.glob(os.path.join(DOCS_DIR, "**", "*.pdf"), recursive=True))

    # Build chunks
    all_chunks = []
    all_metas = []
    for path in tqdm(files, desc="Reading and chunking docs"):
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                text = load_pdf_text(path)
            else:
                text = load_txt_text(path)
            if not text.strip():
                continue
            pieces = split_into_chunks(text, chunk_size=800, overlap=200)
            for i, ch in enumerate(pieces):
                meta = {"source": path, "chunk_index": i}
                all_chunks.append(ch)
                all_metas.append(meta)
        except Exception as e:
            print("Error processing " + path + ": " + str(e))
            continue

    if len(all_chunks) == 0:
        # Create a fallback chunk so the app still runs
        all_chunks = ["No documents indexed yet. Add files to the docs folder and POST /ingest to rebuild."]
        all_metas = [{"source": "__empty__", "chunk_index": 0}]

    model = get_embedding_model()
    embeddings = []
    for ch in tqdm(all_chunks, desc="Embedding chunks"):
        emb = model.encode(ch, show_progress_bar=False)
        embeddings.append(emb)
    embeddings = np.array(embeddings).astype("float32")

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Save
    faiss.write_index(index, FAISS_INDEX)
    with open(CHUNKS_PKL, "wb") as f:
        pickle.dump(all_chunks, f)
    with open(META_PKL, "wb") as f:
        pickle.dump(all_metas, f)

    # Load into globals
    faiss_index = index
    chunks = all_chunks
    metas = all_metas


def retrieve(query: str, k: int = 5) -> List[Dict[str, Any]]:
    global faiss_index, chunks, metas
    if faiss_index is None:
        build_or_load_index(False)
    model = get_embedding_model()
    q_emb = model.encode(query, show_progress_bar=False).astype("float32")
    q_emb = q_emb.reshape(1, -1)
    faiss.normalize_L2(q_emb)
    D, I = faiss_index.search(q_emb, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        results.append({
            "score": float(score),
            "chunk": chunks[idx],
            "meta": metas[idx]
        })
    return results


def answer_question(question: str, k: int = 5) -> Dict[str, Any]:
    retrieved = retrieve(question, k=k)
    qa = get_qa_pipeline()

    # Run extractive QA over each chunk and take the best
    best = None
    for r in retrieved:
        try:
            res = qa(question=question, context=r["chunk"])
            candidate = {
                "answer": res.get("answer", ""),
                "confidence": float(res.get("score", 0.0)),
                "source": r["meta"].get("source", ""),
                "chunk_index": r["meta"].get("chunk_index", -1),
                "retrieval_score": r.get("score", 0.0)
            }
            if best is None or candidate["confidence"] > best["confidence"]:
                best = candidate
        except Exception as e:
            continue

    if best is None and len(retrieved) > 0:
        # Fallback: return the first chunk snippet
        r0 = retrieved[0]
        best = {
            "answer": r0["chunk"][:300],
            "confidence": 0.0,
            "source": r0["meta"].get("source", ""),
            "chunk_index": r0["meta"].get("chunk_index", -1),
            "retrieval_score": r0.get("score", 0.0)
        }

    return {
        "question": question,
        "answer": best.get("answer", "") if best else "",
        "confidence": best.get("confidence", 0.0) if best else 0.0,
        "source": best.get("source", "") if best else "",
        "chunk_index": best.get("chunk_index", -1) if best else -1,
        "retrieval_score": best.get("retrieval_score", 0.0) if best else 0.0,
        "retrieved": [{
            "score": float(r.get("score", 0.0)),
            "source": r["meta"].get("source", ""),
            "chunk_index": r["meta"].get("chunk_index", -1),
            "snippet": r["chunk"][:300]
        } for r in retrieved]
    }

# ----------------------------- Tornado Handlers -----------------------------

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

class AskHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Content-Type", "application/json")

    async def post(self):
        try:
            data = json.loads(self.request.body)
            question = data.get("question", "")
            k = int(data.get("k", 5))
            if not question:
                self.set_status(400)
                self.write(json.dumps({"error": "Missing question"}))
                return
            result = answer_question(question, k=k)
            self.write(json.dumps(result))
        except Exception as e:
            self.set_status(500)
            self.write(json.dumps({"error": str(e)}))

class IngestHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Content-Type", "application/json")

    async def post(self):
        try:
            force = True
            build_or_load_index(force)
            self.write(json.dumps({"status": "ok", "message": "Index rebuilt", "chunks": len(chunks)}))
        except Exception as e:
            self.set_status(500)
            self.write(json.dumps({"error": str(e)}))


def make_app():
    ensure_dirs()
    build_or_load_index(False)
    return tornado.web.Application([
        (r"/", IndexHandler),
        (r"/ask", AskHandler),
        (r"/ingest", IngestHandler),
    ], debug=True)

if __name__ == "__main__":
    app = make_app()
    port = int(os.environ.get("PORT", 8686))
    app.listen(port)
    print("RAG app listening on http://localhost:" + str(port))
    tornado.ioloop.IOLoop.current().start()
