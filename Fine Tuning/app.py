"""
Support Ticket Classifier — Web App
=====================================
Serves the fine-tuned DistilBERT model via a Tornado web server.

Usage:
    python app.py
    Then open http://localhost:8686
"""

import tornado.ioloop
import tornado.web
import json
from transformers import pipeline

MODEL_DIR = "./ticket-classifier"

print("Loading fine-tuned model...")
classifier = pipeline(
    "text-classification",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,
)
print("Model ready!")

# Category metadata for richer UI responses
CATEGORY_META = {
    "Billing": {
        "icon": "💳",
        "description": "Payment, invoices, charges, refunds",
        "color": "blue",
    },
    "Technical": {
        "icon": "🔧",
        "description": "Bugs, errors, app issues, integrations",
        "color": "orange",
    },
    "Delivery": {
        "icon": "📦",
        "description": "Shipping, tracking, couriers",
        "color": "green",
    },
    "Returns": {
        "icon": "↩️",
        "description": "Returns, exchanges, replacements",
        "color": "purple",
    },
}


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")


class ClassifyHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Content-Type", "application/json")

    async def post(self):
        try:
            data = json.loads(self.request.body)
            text = data.get("text", "").strip()

            if not text:
                self.set_status(400)
                self.write(json.dumps({"error": "No text provided"}))
                return

            result = classifier(text)[0]
            label = result["label"]
            confidence = round(result["score"] * 100, 1)
            meta = CATEGORY_META.get(label, {})

            self.write(json.dumps({
                "label": label,
                "confidence": confidence,
                "icon": meta.get("icon", "❓"),
                "description": meta.get("description", ""),
                "color": meta.get("color", "grey"),
            }))

        except Exception as e:
            self.set_status(500)
            self.write(json.dumps({"error": str(e)}))


def make_app():
    return tornado.web.Application([
        (r"/", IndexHandler),
        (r"/classify", ClassifyHandler),
    ], template_path=".")


if __name__ == "__main__":
    app = make_app()
    app.listen(8686)
    print("App running at http://localhost:8686")
    tornado.ioloop.IOLoop.current().start()
