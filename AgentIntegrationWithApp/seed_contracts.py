"""
Seeds the database with 2 pre-reviewed contracts for demo context.
These represent contracts already processed before the demo starts,
so the queue looks lived-in rather than empty.

Your actual PDF files go in the contracts/ folder.
The app registers them automatically when you click "Check for New".

Run once: python3.9 seed_contracts.py
"""

import sqlite3
from datetime import datetime, timedelta
import json

DB_PATH = "contracts.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS contracts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL UNIQUE,
            vendor TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Pending Review',
            review_type TEXT,
            reviewer TEXT,
            reviewed_at TEXT,
            risk_score TEXT,
            missing_clauses TEXT,
            problematic_terms TEXT,
            key_obligations TEXT,
            recommendations TEXT,
            confidence REAL,
            manual_notes TEXT
        )
    """)
    conn.commit()
    conn.close()


# Two pre-reviewed contracts for demo context.
# These don't need real PDFs — they just populate the history.
SEED_DATA = [
    {
        "filename": "Acme_sow.pdf",
        "vendor": "Acme Services",
        "uploaded_at": (datetime.now() - timedelta(days=2)).isoformat(),
        "status": "Manually Cleared",
        "review_type": "Manual",
        "reviewer": "Priya Sharma",
        "reviewed_at": (datetime.now() - timedelta(days=1, hours=3)).isoformat(),
        "risk_score": "Low",
        "missing_clauses": json.dumps([]),
        "problematic_terms": json.dumps([]),
        "key_obligations": json.dumps([
            "Consulting services delivered per Schedule A",
            "Payment within Net 28 days",
            "30 days notice for termination"
        ]),
        "recommendations": "",
        "confidence": None,
        "manual_notes": "Standard MSA. All clauses present. Payment terms within threshold."
    },
    {
        "filename": "Varaha Tech.pdf",
        "vendor": "Varaha Technology Group",
        "uploaded_at": (datetime.now() - timedelta(days=3)).isoformat(),
        "status": "Escalated for Legal Review",
        "review_type": "Manual",
        "reviewer": "Rahul Menon",
        "reviewed_at": (datetime.now() - timedelta(days=2)).isoformat(),
        "risk_score": "High",
        "missing_clauses": json.dumps(["Indemnification", "Intellectual Property Ownership"]),
        "problematic_terms": json.dumps(["Unlimited liability clause", "Auto-renewal without notice"]),
        "key_obligations": json.dumps([
            "Data analytics platform access",
            "Payment within 60 days (non-compliant)"
        ]),
        "recommendations": "Unlimited liability clause must be removed. IP ownership undefined. Payment terms 2x policy threshold. Do not sign without legal review.",
        "confidence": None,
        "manual_notes": "Unlimited liability clause found. Foreign jurisdiction. Needs legal sign-off."
    }
]


def seed():
    init_db()
    conn = sqlite3.connect(DB_PATH)

    for c in SEED_DATA:
        existing = conn.execute(
            "SELECT id FROM contracts WHERE filename=?", (c["filename"],)
        ).fetchone()
        if existing:
            print(f"  Skipping {c['filename']} — already exists")
            continue

        conn.execute("""
            INSERT INTO contracts (
                filename, vendor, uploaded_at, status,
                review_type, reviewer, reviewed_at, risk_score,
                missing_clauses, problematic_terms, key_obligations,
                recommendations, confidence, manual_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            c["filename"], c["vendor"], c["uploaded_at"], c["status"],
            c["review_type"], c["reviewer"], c["reviewed_at"], c["risk_score"],
            c["missing_clauses"], c["problematic_terms"], c["key_obligations"],
            c["recommendations"], c["confidence"], c["manual_notes"]
        ))
        print(f"  Added: {c['filename']} — {c['status']}")

    conn.commit()
    conn.close()
    print("\nDone. Now place your PDF files in the contracts/ folder")
    print("and click 'Check for New' in the app to register them.")


if __name__ == "__main__":
    seed()
