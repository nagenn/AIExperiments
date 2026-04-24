import os
import json
import sqlite3
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

DB_PATH = "contracts.db"
CONTRACTS_FOLDER = "contracts"


# ----------------------------
# DB SETUP
# ----------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
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


init_db()


# ----------------------------
# MODELS
# ----------------------------
class ManualReview(BaseModel):
    reviewer: str
    limitation_of_liability: bool
    indemnification: bool
    termination_rights: bool
    data_protection: bool
    ip_ownership: bool
    payment_terms_ok: bool
    no_prohibited_terms: bool
    risk_score: str
    notes: Optional[str] = ""


class AgentReview(BaseModel):
    risk_score: str
    missing_clauses: List[str]
    problematic_terms: List[str]
    key_obligations: List[str]
    recommendations: str
    confidence: float


# ----------------------------
# HELPERS
# ----------------------------
def status_from_risk(risk: str) -> str:
    if risk == "High":
        return "Escalated for Legal Review"
    elif risk == "Medium":
        return "Flagged — Needs Attention"
    else:
        return "Agent Cleared"


def derive_vendor(filename: str) -> str:
    """
    Derives a readable vendor name from a filename.
    e.g. 'nexus_consulting_sow.pdf' -> 'Nexus Consulting'
    """
    name = filename.replace(".pdf", "").replace("_", " ").replace("-", " ")
    # Drop common suffixes
    for suffix in ["sow", "msa", "agreement", "contract", "services", "vendor"]:
        name = name.replace(f" {suffix}", "").replace(suffix, "")
    return name.strip().title()


# ----------------------------
# SCAN CONTRACTS FOLDER
# ----------------------------
def scan_contracts_folder():
    """
    Reads all PDFs from the contracts/ folder.
    Registers any new ones in the DB. Does not touch existing records.
    """
    if not os.path.exists(CONTRACTS_FOLDER):
        return []

    pdf_files = [f for f in os.listdir(CONTRACTS_FOLDER) if f.lower().endswith(".pdf")]
    conn = get_db()
    added = []

    for filename in sorted(pdf_files):
        existing = conn.execute(
            "SELECT id FROM contracts WHERE filename=?", (filename,)
        ).fetchone()
        if not existing:
            vendor = derive_vendor(filename)
            conn.execute("""
                INSERT INTO contracts (filename, vendor, uploaded_at, status)
                VALUES (?, ?, ?, 'Pending Review')
            """, (filename, vendor, datetime.now().isoformat()))
            added.append(filename)

    conn.commit()
    conn.close()
    return added


# ----------------------------
# ROUTES
# ----------------------------
@app.get("/")
def serve_index():
    return FileResponse(
        "static/index.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@app.post("/api/contracts/scan")
def scan_contracts():
    if not os.path.exists(CONTRACTS_FOLDER):
        return {"added": [], "count": 0, "message": "contracts/ folder not found. Please create it."}
    added = scan_contracts_folder()
    return {
        "added": added,
        "count": len(added),
        "message": f"{len(added)} new contract(s) added." if added else "No new contracts found."
    }


@app.get("/api/contracts")
def list_contracts():
    conn = get_db()
    rows = conn.execute("SELECT * FROM contracts ORDER BY uploaded_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/contracts/{contract_id}")
def get_contract(contract_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM contracts WHERE id=?", (contract_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Contract not found")
    return dict(row)


@app.post("/api/contracts/{contract_id}/manual-review")
def manual_review(contract_id: int, review: ManualReview):
    if review.risk_score == "High":
        new_status = "Escalated for Legal Review"
    elif review.risk_score == "Medium":
        new_status = "Flagged — Needs Attention"
    else:
        new_status = "Manually Cleared"

    conn = get_db()
    conn.execute("""
        UPDATE contracts SET
            status=?, review_type='Manual', reviewer=?, reviewed_at=?,
            risk_score=?, manual_notes=?
        WHERE id=?
    """, (new_status, review.reviewer, datetime.now().isoformat(),
          review.risk_score, review.notes, contract_id))
    conn.commit()
    conn.close()
    return {"status": new_status}


@app.post("/api/contracts/{contract_id}/analyse")
def analyse_contract(contract_id: int):
    """
    Triggers LLM analysis for a contract.
    Reads the PDF from the contracts/ folder, runs analysis, saves result.
    """
    conn = get_db()
    row = conn.execute("SELECT * FROM contracts WHERE id=?", (contract_id,)).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Contract not found")

    contract = dict(row)
    pdf_path = os.path.join(CONTRACTS_FOLDER, contract["filename"])

    if not os.path.exists(pdf_path):
        raise HTTPException(
            status_code=400,
            detail=f"PDF not found at {pdf_path}. Make sure the file is in the contracts/ folder."
        )

    # Import here so .env is already loaded
    from analyze_contract import run_analysis
    result = run_analysis(pdf_path)

    new_status = status_from_risk(result.get("risk_score", "Unknown"))

    conn = get_db()
    conn.execute("""
        UPDATE contracts SET
            status=?, review_type='Agent', reviewer='AI Agent', reviewed_at=?,
            risk_score=?, missing_clauses=?, problematic_terms=?,
            key_obligations=?, recommendations=?, confidence=?
        WHERE id=?
    """, (
        new_status,
        datetime.now().isoformat(),
        result.get("risk_score", "Unknown"),
        json.dumps(result.get("missing_clauses", [])),
        json.dumps(result.get("problematic_terms", [])),
        json.dumps(result.get("key_obligations", [])),
        result.get("recommendations", ""),
        result.get("confidence", 0.0),
        contract_id
    ))
    conn.commit()
    conn.close()

    return {"status": new_status, "result": result}


@app.post("/api/contracts/{contract_id}/agent-review")
def agent_review_post(contract_id: int, result: AgentReview):
    """Legacy endpoint — kept for compatibility."""
    new_status = status_from_risk(result.risk_score)
    conn = get_db()
    conn.execute("""
        UPDATE contracts SET
            status=?, review_type='Agent', reviewer='AI Agent', reviewed_at=?,
            risk_score=?, missing_clauses=?, problematic_terms=?,
            key_obligations=?, recommendations=?, confidence=?
        WHERE id=?
    """, (
        new_status,
        datetime.now().isoformat(),
        result.risk_score,
        json.dumps(result.missing_clauses),
        json.dumps(result.problematic_terms),
        json.dumps(result.key_obligations),
        result.recommendations,
        result.confidence,
        contract_id
    ))
    conn.commit()
    conn.close()
    return {"status": new_status}


@app.post("/api/contracts/{contract_id}/reset")
def reset_contract(contract_id: int):
    conn = get_db()
    conn.execute("""
        UPDATE contracts SET
            status='Pending Review', review_type=NULL, reviewer=NULL,
            reviewed_at=NULL, risk_score=NULL, missing_clauses=NULL,
            problematic_terms=NULL, key_obligations=NULL,
            recommendations=NULL, confidence=NULL, manual_notes=NULL
        WHERE id=?
    """, (contract_id,))
    conn.commit()
    conn.close()
    return {"status": "reset"}


app.mount("/static", StaticFiles(directory="static"), name="static")
