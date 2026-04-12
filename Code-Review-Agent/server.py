"""
server.py — Tornado webhook server.

Listens for GitHub PR webhook events, validates the signature,
and hands off to the agent asynchronously.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import tornado.ioloop
import tornado.web

import config
from agent import run_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PORT = 8080


def verify_signature(payload_bytes: bytes, signature_header: str) -> bool:
    """Verify GitHub's HMAC-SHA256 webhook signature."""
    if not signature_header:
        return False
    expected = "sha256=" + hmac.new(
        config.GITHUB_WEBHOOK_SECRET.encode(),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header)


class WebhookHandler(tornado.web.RequestHandler):

    async def post(self):
        # 1. Validate signature
        sig = self.request.headers.get("X-Hub-Signature-256", "")
        if not verify_signature(self.request.body, sig):
            log.warning("❌ Invalid webhook signature — request rejected")
            self.set_status(401)
            self.write({"error": "Invalid signature"})
            return

        # 2. Handle ping event from GitHub
        event = self.request.headers.get("X-GitHub-Event", "")
        if event == "ping":
            log.info("🏓 GitHub ping received — webhook configured correctly!")
            self.write({"status": "pong"})
            return
        log.info(f"📨 Event: {event}, Body length: {len(self.request.body)}")
        log.info(f"📨 Body preview: {self.request.body[:100]}")

        # 3. Parse payload
        #payload = json.loads(self.request.body)
        payload = json.loads(self.request.body.decode("utf-8"))

        log.info(f"📨 Received GitHub event: {event}")

        # 4. Only handle PR open / reopen / synchronize events
        if event == "pull_request":
            action = payload.get("action", "")
            if action in ("opened", "reopened", "synchronize"):
                pr_number = payload["pull_request"]["number"]
                pr_meta = payload["pull_request"]
                log.info(f"✅ PR #{pr_number} ({action}) — spawning agent")

                asyncio.get_event_loop().run_in_executor(
                    None, run_agent, pr_number, pr_meta
                )

                self.write({"status": "review started", "pr": pr_number})
            else:
                log.info(f"   Skipping PR action: {action}")
                self.write({"status": "ignored", "action": action})
        else:
            log.info(f"   Skipping event: {event}")
            self.write({"status": "ignored", "event": event})


class HealthHandler(tornado.web.RequestHandler):
    def get(self):
        self.write({"status": "ok", "model": config.OPENAI_MODEL})


def make_app():
    return tornado.web.Application([
        (r"/webhook", WebhookHandler),
        (r"/health", HealthHandler),
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(PORT)
    log.info(f"🌐 Tornado server listening on port 8080")
    log.info(f"🔗 Webhook endpoint: http://localhost:{PORT}/webhook")
    log.info(f"💡 Health check:     http://localhost:{PORT}/health")
    log.info(f"🤖 Model: {config.OPENAI_MODEL}")
    tornado.ioloop.IOLoop.current().start()