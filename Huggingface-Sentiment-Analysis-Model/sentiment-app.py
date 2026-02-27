import tornado.ioloop
import tornado.web
import json
from transformers import pipeline

# Load the sentiment analysis model at startup
print("Loading sentiment model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)
print("Model ready!")

# Map model labels to human-friendly names
LABEL_MAP = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral":  "Neutral",
}

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

class SentimentHandler(tornado.web.RequestHandler):
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

            result = sentiment_pipeline(text)[0]

            label = result["label"].lower()
            friendly_label = LABEL_MAP.get(label, result["label"])
            confidence = round(result["score"] * 100, 1)

            self.write(json.dumps({
                "text": text,
                "label": friendly_label,
                "confidence": confidence,
            }))

        except Exception as e:
            self.set_status(500)
            self.write(json.dumps({"error": str(e)}))


def make_app():
    return tornado.web.Application([
        (r"/", IndexHandler),
        (r"/analyse", SentimentHandler),
    ], template_path=".")


if __name__ == "__main__":
    app = make_app()
    app.listen(8686)
    print("Sentiment analyser running at http://localhost:8686")
    tornado.ioloop.IOLoop.current().start()
