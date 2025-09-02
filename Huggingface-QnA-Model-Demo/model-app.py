import tornado.ioloop
import tornado.web
import json
from transformers import pipeline

# Load Hugging Face QA pipeline
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Load context from a plain text file
def load_context(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading context file: {e}")
        return ""

# Load context once at startup
context = load_context("freshbite_faq.txt")

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")  # This looks for index.html in your current directory

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

            result = qa_pipeline(question=question, context=context)
            self.write(json.dumps({
                "question": question,
                "answer": result["answer"],
                "confidence": round(result["score"], 2)
            }))
        except Exception as e:
            self.set_status(500)
            self.write(json.dumps({"error": str(e)}))

#def make_app():
 #   return tornado.web.Application([
  #      (r"/ask", QnAHandler),
   # ])

def make_app():
    return tornado.web.Application([
        (r"/", IndexHandler),        # Serve the HTML
        (r"/ask", QnAHandler),       # Serve the QA API
    ], template_path=".")


if __name__ == "__main__":
    app = make_app()
    app.listen(8686)
    print("Tornado QA API running at http://localhost:8686/ask")
    tornado.ioloop.IOLoop.current().start()
