"""
chroma_mcp_server.py
--------------------
ChromaDB Memory MCP Server — Terminal 6

Single-process owner of ChromaDB. Eliminates the HNSW file-locking bug
that occurred when multiple agent processes opened PersistentClient simultaneously.

Runs as a persistent SSE (HTTP) server so Terminal 6 stays visible during
the demo — audience can see it receive calls from each agent in real time.

Tools exposed:
  • query_vendor_history   — semantic + fuzzy vendor search
  • query_similar_clauses  — semantic clause search by type
  • store_review           — store completed contract review + individual clauses
  • get_stats              — contracts count, clause count, vendors list

Transport: SSE on http://localhost:8765
Run: python3.9 chroma_mcp_server.py

Install: pip install "mcp[cli]" chromadb openai
"""

import json
import os
import logging
from datetime import datetime

from mcp.server.fastmcp import FastMCP

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    filename="memory_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def _log(msg):
    logging.debug(msg)


# ── ChromaDB setup ────────────────────────────────────────────────────────────
# PersistentClient is safe here because this MCP server is the ONLY process
# that will ever open it. No cross-process file locking can occur.

DB_PATH              = "./contract_memory_db"
COLLECTION_CONTRACTS = "contracts"
COLLECTION_CLAUSES   = "clauses"

if CHROMA_AVAILABLE:
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    _client = chromadb.PersistentClient(path=DB_PATH)
    _log("ChromaDB PersistentClient initialised at startup.")
else:
    _client = None
    _log("WARNING: chromadb not installed — all tools will return errors.")


def _col(name):
    if not CHROMA_AVAILABLE:
        raise RuntimeError("ChromaDB not available. Run: pip install chromadb")
    return _client.get_or_create_collection(
        name,
        embedding_function=embedding_function
    )


# ── Vendor fuzzy matching ─────────────────────────────────────────────────────

def _vendor_words(name):
    """Strip legal suffixes and short words; return significant word set."""
    skip = {"the", "and", "for", "pte", "ltd", "llc", "inc",
            "corp", "co", "plc", "gmbh", "bv", "ag", "sas"}
    return set(
        w.lower()
        for w in name.replace(".", "").replace(",", "").split()
        if len(w) > 3 and w.lower() not in skip
    )


def _vendor_matches(query_name, stored_name):
    """True if any significant word overlaps between the two vendor names."""
    query_words  = _vendor_words(query_name)
    stored_words = _vendor_words(stored_name)
    if not query_words or not stored_words:
        return False
    return bool(query_words & stored_words)


# ── Result formatter ──────────────────────────────────────────────────────────

def _fmt(results):
    out = []
    if not results or not results.get("documents"):
        return out
    docs  = results["documents"][0] if results.get("documents") else []
    metas = results["metadatas"][0]  if results.get("metadatas") else []
    dists = results["distances"][0]  if results.get("distances") else []
    for i, doc in enumerate(docs):
        out.append({
            "document":   doc,
            "metadata":   metas[i] if i < len(metas) else {},
            "similarity": round(1 - dists[i], 3) if i < len(dists) else None
        })
    return out


# ── MCP server ────────────────────────────────────────────────────────────────

MCP_HOST = "localhost"
MCP_PORT = 8765

mcp = FastMCP("chroma-memory", host=MCP_HOST, port=MCP_PORT)


@mcp.tool()
def query_vendor_history(vendor_name: str, n: int = 5) -> str:
    """
    Semantic search for prior contracts with a named vendor.

    Two-step: exact where-filter first, then semantic + Python fuzzy fallback.
    Fuzzy matching strips legal suffixes (Pte, Ltd, Inc etc.) and matches on
    significant words — so 'DataSync Solutions' matches
    'DataSync Solutions Pte. Ltd.'

    Returns JSON list of dicts: [{document, metadata, similarity}, ...]
    """
    if not CHROMA_AVAILABLE:
        return json.dumps({"error": "ChromaDB not available."})

    try:
        col   = _col(COLLECTION_CONTRACTS)
        total = col.count()
        _log(f"query_vendor_history: vendor='{vendor_name}' total_docs={total}")

        if total == 0:
            _log("query_vendor_history: collection empty — returning []")
            return json.dumps([])

        safe_n = min(n, total)

        # Step 1: exact match via metadata filter
        try:
            results   = col.query(
                query_texts=[f"vendor {vendor_name}"],
                n_results=safe_n,
                where={"vendor": vendor_name}
            )
            formatted = _fmt(results)
            _log(f"query_vendor_history: exact match returned {len(formatted)} result(s)")
            if formatted:
                return json.dumps(formatted)
        except Exception as e:
            _log(f"query_vendor_history: exact match EXCEPTION — {e}")

        # Step 2: semantic search + Python fuzzy matching
        safe_n2 = min(n * 3, total)
        _log(f"query_vendor_history: semantic fallback n_results={safe_n2}")
        try:
            results   = col.query(
                query_texts=[f"vendor contract {vendor_name}"],
                n_results=safe_n2
            )
            formatted = _fmt(results)
            _log(f"query_vendor_history: semantic fallback returned {len(formatted)} result(s)")

            if not formatted:
                _log("query_vendor_history: semantic fallback empty — returning []")
                return json.dumps([])

            matches = [
                r for r in formatted
                if _vendor_matches(vendor_name, r.get("metadata", {}).get("vendor", ""))
            ]
            all_vendors = [r.get("metadata", {}).get("vendor", "") for r in formatted]
            _log(f"query_vendor_history: fuzzy found {len(matches)} match(es) "
                 f"from vendors: {all_vendors}")
            return json.dumps(matches[:n])

        except Exception as e:
            _log(f"query_vendor_history: semantic fallback EXCEPTION — {e}")
            return json.dumps([])

    except Exception as e:
        _log(f"query_vendor_history: outer EXCEPTION — {e}")
        return json.dumps({"error": str(e)})


@mcp.tool()
def query_similar_clauses(clause_text: str, clause_type: str, n: int = 5) -> str:
    """
    Find similar clauses across all past contracts.

    clause_type must be one of: liability_cap, payment_terms, data_rights, sla

    Two-step: exact type filter first, then semantic fallback filtered in Python.

    Returns JSON list of dicts: [{document, metadata, similarity}, ...]
    """
    if not CHROMA_AVAILABLE:
        return json.dumps({"error": "ChromaDB not available."})

    valid_types = {"liability_cap", "payment_terms", "data_rights", "sla"}
    if clause_type not in valid_types:
        return json.dumps({"error": f"Invalid clause_type. Must be one of: {valid_types}"})

    try:
        col   = _col(COLLECTION_CLAUSES)
        total = col.count()
        _log(f"query_similar_clauses: type='{clause_type}' total_docs={total}")

        if total == 0:
            _log("query_similar_clauses: collection empty — returning []")
            return json.dumps([])

        safe_n = min(n, total)

        # Step 1: exact type match via metadata filter
        try:
            results   = col.query(
                query_texts=[clause_text],
                n_results=safe_n,
                where={"type": clause_type}
            )
            formatted = _fmt(results)
            _log(f"query_similar_clauses: exact match returned {len(formatted)} result(s)")
            if formatted:
                return json.dumps(formatted)
        except Exception as e:
            _log(f"query_similar_clauses: exact match EXCEPTION — {e}")

        # Step 2: semantic fallback, filter by type in Python
        safe_n2   = min(n * 2, total)
        results   = col.query(
            query_texts=[clause_text],
            n_results=safe_n2
        )
        formatted = _fmt(results)
        matches   = [
            r for r in formatted
            if r.get("metadata", {}).get("type", "") == clause_type
        ]
        _log(f"query_similar_clauses: fallback found {len(matches)} match(es)")
        return json.dumps(matches[:n])

    except Exception as e:
        _log(f"query_similar_clauses: EXCEPTION — {e}")
        return json.dumps({"error": str(e)})


@mcp.tool()
def store_review(
    contract_id:   str,
    vendor_name:   str,
    contract_text: str,
    findings_json: str,
    verdict:       str
) -> str:
    """
    Store a completed contract review in ChromaDB.

    findings_json: JSON string of the merged extraction + risk findings dict.
    Also stores individual clauses (liability_cap, payment_terms,
    data_rights, sla) into the clauses collection.

    Because this MCP server is the ONLY process touching ChromaDB,
    no client reset is needed — the HNSW index is always consistent.

    Returns a JSON status object.
    """
    if not CHROMA_AVAILABLE:
        return json.dumps({"error": "ChromaDB not available."})

    try:
        findings = json.loads(findings_json) if isinstance(findings_json, str) \
                   else findings_json
    except Exception as e:
        return json.dumps({"error": f"Could not parse findings_json: {e}"})

    try:
        col  = _col(COLLECTION_CONTRACTS)
        meta = {
            "vendor":        vendor_name,
            "contract_id":   contract_id,
            "review_date":   datetime.now().isoformat(),
            "verdict":       verdict,
            "liability_cap": str(findings.get("liability_cap", "")),
            "payment_terms": str(findings.get("payment_due_days", "")),
            "governing_law": str(findings.get("governing_law", "")),
            "overall_risk":  str(findings.get("overall_risk", "")),
        }
        doc = (
            f"Vendor: {vendor_name} | Verdict: {verdict} | "
            f"Liability: {meta['liability_cap']} | "
            f"Payment: {meta['payment_terms']} days | "
            f"Risk: {meta['overall_risk']} | "
            f"Reviewed: {meta['review_date'][:10]} | "
            f"Findings: {json.dumps(findings)[:600]}"
        )
        col.upsert(documents=[doc], metadatas=[meta], ids=[contract_id])
        contracts_count = col.count()
        _log(f"store_review: upserted '{contract_id}'. contracts total={contracts_count}")

        # Store individual clauses
        clauses_stored = _store_clauses(contract_id, vendor_name, findings, verdict)

        return json.dumps({
            "status":          "stored",
            "contract_id":     contract_id,
            "contracts_total": contracts_count,
            "clauses_stored":  clauses_stored
        })

    except Exception as e:
        _log(f"store_review: EXCEPTION — {e}")
        return json.dumps({"error": str(e)})


def _store_clauses(contract_id, vendor_name, findings, verdict):
    """Store individual clause types into the clauses collection."""
    col        = _col(COLLECTION_CLAUSES)
    stored_any = 0

    for field, clause_type in [
        ("liability_cap",     "liability_cap"),
        ("payment_due_days",  "payment_terms"),
        ("data_usage_rights", "data_rights"),
        ("uptime_sla",        "sla"),
    ]:
        value = str(findings.get(field, "")).strip()
        if value:
            col.upsert(
                documents=[value],
                metadatas=[{
                    "type":        clause_type,
                    "vendor":      vendor_name,
                    "contract_id": contract_id,
                    "verdict":     verdict
                }],
                ids=[f"{contract_id}_{clause_type}"]
            )
            stored_any += 1
            _log(f"_store_clauses: stored {clause_type} for {vendor_name}")

    return stored_any


@mcp.tool()
def get_stats() -> str:
    """
    Return memory statistics: contracts count, clauses count, vendors list.
    Used by demo.py to show memory status in the menu header.

    Returns JSON: {contracts: int, clauses: int, vendors: [str]}
    """
    stats = {"contracts": 0, "clauses": 0, "vendors": []}

    if not CHROMA_AVAILABLE:
        return json.dumps(stats)

    try:
        data = _col(COLLECTION_CONTRACTS).get()
        if data and data.get("metadatas"):
            stats["contracts"] = len(data["metadatas"])
            vendor_set = set()
            for m in data["metadatas"]:
                if m.get("vendor"):
                    vendor_set.add(m["vendor"])
            stats["vendors"] = list(vendor_set)
    except Exception as e:
        _log(f"get_stats: contracts EXCEPTION — {e}")

    try:
        data = _col(COLLECTION_CLAUSES).get()
        if data and data.get("ids"):
            stats["clauses"] = len(data["ids"])
    except Exception as e:
        _log(f"get_stats: clauses EXCEPTION — {e}")

    _log(f"get_stats: {stats}")
    return json.dumps(stats)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n  ╔══════════════════════════════════════════════════════╗")
    print(f"  ║   CHROMA MEMORY MCP SERVER — Terminal 6              ║")
    print(f"  ╚══════════════════════════════════════════════════════╝")
    print(f"\n  Listening on http://{MCP_HOST}:{MCP_PORT}")
    print(f"  Tools: query_vendor_history | query_similar_clauses")
    print(f"         store_review | get_stats")
    print(f"\n  Waiting for agent calls...\n")
    _log(f"chroma_mcp_server: starting SSE on {MCP_HOST}:{MCP_PORT}")
    mcp.run(transport="sse")
