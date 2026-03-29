"""
api.py  —  FastAPI REST wrapper over support_history.db

Run with:
    uvicorn api:app --reload --port 8000

Endpoints:
    GET /customers/{customer_id}            — customer profile
    GET /customers/{customer_id}/history    — full interaction history
    GET /customers/{customer_id}/summary    — aggregated stats (purchases, returns, etc.)
    GET /health                             — liveness check
"""

import os
import sqlite3
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "support_history.db")

app = FastAPI(title="Customer History API", version="1.0")


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def row_to_dict(row) -> dict:
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "db": os.path.exists(DB_PATH)}


@app.get("/customers/{customer_id}")
def get_customer(customer_id: str):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    return row_to_dict(row)


@app.get("/customers/{customer_id}/history")
def get_history(customer_id: str, limit: int = 50):
    with get_db() as conn:
        # Check customer exists
        customer = conn.execute(
            "SELECT customer_id FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()
        if not customer:
            raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

        rows = conn.execute(
            """SELECT * FROM interactions
               WHERE customer_id = ?
               ORDER BY date DESC
               LIMIT ?""",
            (customer_id, limit),
        ).fetchall()

    return {"customer_id": customer_id, "interactions": [row_to_dict(r) for r in rows]}


@app.get("/customers/{customer_id}/summary")
def get_summary(customer_id: str):
    with get_db() as conn:
        customer = conn.execute(
            "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()
        if not customer:
            raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

        stats = conn.execute(
            """SELECT
                COUNT(*) as total_interactions,
                SUM(CASE WHEN type = 'purchase' THEN 1 ELSE 0 END) as total_purchases,
                SUM(CASE WHEN type = 'return_request' THEN 1 ELSE 0 END) as total_returns,
                SUM(CASE WHEN type = 'billing_dispute' THEN 1 ELSE 0 END) as billing_disputes,
                SUM(CASE WHEN type = 'technical_issue' THEN 1 ELSE 0 END) as technical_issues,
                SUM(CASE WHEN type = 'cancellation_request' THEN 1 ELSE 0 END) as cancellation_requests,
                SUM(CASE WHEN resolution = 'escalated' THEN 1 ELSE 0 END) as times_escalated,
                SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) as negative_interactions,
                SUM(CASE WHEN amount_usd > 0 THEN amount_usd ELSE 0 END) as total_spend_usd,
                MAX(date) as last_interaction_date
               FROM interactions
               WHERE customer_id = ?""",
            (customer_id,),
        ).fetchone()

        # Most recent support ticket
        last_ticket = conn.execute(
            """SELECT type, description, resolution, date
               FROM interactions
               WHERE customer_id = ? AND type != 'purchase'
               ORDER BY date DESC LIMIT 1""",
            (customer_id,),
        ).fetchone()

    return {
        "customer": row_to_dict(customer),
        "stats": row_to_dict(stats),
        "last_support_interaction": row_to_dict(last_ticket),
    }
