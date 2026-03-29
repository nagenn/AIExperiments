"""
seed_db.py  —  Generate synthetic data with GPT and load into SQLite.

Run once before the demo:
    python seed_db.py

Creates support_history.db with two tables:
    customers     — 50 customers with tier, contract value, SLA tier, etc.
    interactions  — 100+ rows per customer: purchases, returns, support tickets
"""

import json
import os
import sqlite3

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "support_history.db")


# ---------------------------------------------------------------------------
# GPT generation helpers
# ---------------------------------------------------------------------------

def gpt(prompt: str, max_tokens: int = 4000) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.8,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences GPT sometimes wraps JSON in
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]
    return raw.strip()


def generate_customers() -> list[dict]:
    """Generate 50 customers in two batches of 25 to avoid token truncation."""
    all_customers = []

    batches = [
        ("CUST-001", "CUST-025", 25, "4 Enterprise/Platinum, 6 Business/Gold, 10 Standard/Silver, 5 Trial/Bronze"),
        ("CUST-026", "CUST-050", 25, "4 Enterprise/Platinum, 6 Business/Gold, 10 Standard/Silver, 5 Trial/Bronze"),
    ]

    for start_id, end_id, count, distribution in batches:
        print(f"  Generating customers {start_id} to {end_id}...")
        raw = gpt(f"""
Generate exactly {count} realistic B2B/B2C software customers as a JSON array.
Each object must have these exact keys:
  customer_id     (string, from "{start_id}" through "{end_id}")
  name            (realistic full name, mix of cultures)
  company         (realistic company name)
  email           (realistic email)
  tier            (one of: "Enterprise", "Business", "Standard", "Trial")
  contract_value  (annual USD integer: Enterprise 50000-200000, Business 10000-49999, Standard 1000-9999, Trial 0)
  join_date       (ISO date string between 2019-01-01 and 2023-12-31)
  country         (mix of US, UK, Germany, India, Singapore, Australia, Canada, Brazil)
  sla_tier        (one of: "Platinum", "Gold", "Silver", "Bronze" — correlate with tier)

Distribution for this batch: {distribution}

Return ONLY a valid JSON array. No markdown, no explanation, no code fences.
""", max_tokens=3000)

        try:
            batch = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  ❌ JSON parse error in customer batch {start_id}-{end_id}: {e}")
            print(f"  Raw response (last 200 chars): ...{raw[-200:]}")
            raise
        all_customers.extend(batch)
        print(f"  ✅ {len(batch)} customers generated")

    return all_customers


def generate_interactions(customers: list[dict]) -> list[dict]:
    """Generate interactions in batches of 5 customers to stay safely within token limits."""
    all_interactions = []
    batch_size = 5
    batches = [customers[i:i+batch_size] for i in range(0, len(customers), batch_size)]

    for idx, batch in enumerate(batches):
        print(f"  Generating interactions batch {idx+1}/{len(batches)} ({len(batch)} customers)...")
        customer_summary = json.dumps(
            [{"customer_id": c["customer_id"], "name": c["name"],
              "tier": c["tier"], "join_date": c["join_date"]} for c in batch]
        )
        start_id = idx * batch_size * 8 + 1
        raw = gpt(f"""
For each customer below, generate exactly 6 to 8 interaction records (no more).
Customers: {customer_summary}

Each interaction must have these exact keys:
  interaction_id   (unique string, format "INT-XXXXX" with 5 digits, start from {start_id:05d}, increment by 1)
  customer_id      (matching the customer)
  date             (ISO date string between join_date and 2024-12-31, chronologically plausible)
  type             (one of: "purchase", "return_request", "billing_dispute", "technical_issue",
                   "account_access", "delivery_complaint", "feature_request",
                   "cancellation_request", "data_privacy_request", "general_enquiry")
  description      (1 sentence realistic description of the interaction)
  amount_usd       (integer: purchases 100-5000, returns negative of purchase, others 0)
  resolution       (one of: "resolved", "escalated", "refunded", "denied", "pending", "cancelled")
  sentiment        (one of: "positive", "neutral", "negative")

Rules:
- Each customer must have at least 2 purchases
- Vary the types — not all purchases
- Max 1 pending resolution per customer
- Keep descriptions short — 1 sentence only

Return ONLY a valid JSON array. No markdown, no explanation, no code fences. Close all brackets properly.
""", max_tokens=4000)

        try:
            batch_interactions = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  ❌ JSON parse error in interactions batch {idx+1}: {e}")
            print(f"  Raw response (last 300 chars): ...{raw[-300:]}")
            raise

        all_interactions.extend(batch_interactions)
        print(f"  ✅ {len(batch_interactions)} interactions generated for batch {idx+1}")

    return all_interactions


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

def create_db(customers: list[dict], interactions: list[dict]):
    print(f"\nCreating SQLite database at {DB_PATH}...")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Customers table
    cur.execute("""
        CREATE TABLE customers (
            customer_id     TEXT PRIMARY KEY,
            name            TEXT,
            company         TEXT,
            email           TEXT,
            tier            TEXT,
            contract_value  INTEGER,
            join_date       TEXT,
            country         TEXT,
            sla_tier        TEXT
        )
    """)

    # Interactions table
    cur.execute("""
        CREATE TABLE interactions (
            interaction_id  TEXT PRIMARY KEY,
            customer_id     TEXT,
            date            TEXT,
            type            TEXT,
            description     TEXT,
            amount_usd      INTEGER,
            resolution      TEXT,
            sentiment       TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        )
    """)

    # Insert customers
    for c in customers:
        cur.execute("""
            INSERT INTO customers VALUES (
                :customer_id, :name, :company, :email,
                :tier, :contract_value, :join_date, :country, :sla_tier
            )
        """, c)

    # Insert interactions
    for i in interactions:
        cur.execute("""
            INSERT OR IGNORE INTO interactions VALUES (
                :interaction_id, :customer_id, :date, :type,
                :description, :amount_usd, :resolution, :sentiment
            )
        """, i)

    conn.commit()
    conn.close()

    print(f"  ✅ {len(customers)} customers inserted")
    print(f"  ✅ {len(interactions)} interactions inserted")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Support Triage Agent — DB Seed Script")
    print("=" * 60)

    customers = generate_customers()
    print(f"  ✅ {len(customers)} customers generated")

    print("Generating interaction history with GPT...")
    interactions = generate_interactions(customers)
    print(f"  ✅ {len(interactions)} interactions generated")

    create_db(customers, interactions)

    print("\n✅ Database ready. You can now run: uvicorn api:app --reload")
    print("=" * 60)
