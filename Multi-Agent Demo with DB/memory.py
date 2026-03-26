"""
memory.py
---------
Organisational memory using ChromaDB (local, no cloud needed).
Uses OpenAI embeddings — no local model download required.

Install: pip install chromadb openai
"""

import json
import os
import logging
from datetime import datetime

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

DB_PATH              = "./contract_memory_db"
COLLECTION_CONTRACTS = "contracts"
COLLECTION_CLAUSES   = "clauses"


# ── Debug logging ─────────────────────────────────────────────────────────────

logging.basicConfig(
    filename="memory_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def _log(msg):
    logging.debug(msg)


# ── OpenAI embedding function ─────────────────────────────────────────────────

if CHROMA_AVAILABLE:
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )


# ── Singleton ChromaDB client ─────────────────────────────────────────────────

_client = None

def _get_client():
    global _client
    if _client is None:
        _log(f"Creating new PersistentClient at {DB_PATH}")
        _client = chromadb.PersistentClient(path=DB_PATH)
    else:
        _log("Reusing existing PersistentClient")
    return _client


def _col(name):
    if not CHROMA_AVAILABLE:
        raise RuntimeError("ChromaDB not available")
    return _get_client().get_or_create_collection(
        name,
        embedding_function=embedding_function
    )


# ── Store ─────────────────────────────────────────────────────────────────────

def store_review(contract_id, vendor_name, contract_text, findings, verdict):
    if not CHROMA_AVAILABLE:
        return

    _log(f"store_review: contract_id={contract_id} vendor={vendor_name} "
         f"verdict={verdict}")

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
    _log(f"store_review: upsert complete. count={col.count()}")

    # Reset singleton to force index flush before next cross-process query
    global _client
    _client = None
    _log("store_review: client reset to force index flush")

    _store_clauses(contract_id, vendor_name, findings)


def _store_clauses(contract_id, vendor_name, findings):
    if not CHROMA_AVAILABLE:
        return

    col        = _col(COLLECTION_CLAUSES)
    stored_any = False

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
                metadatas={
                    "type":        clause_type,
                    "vendor":      vendor_name,
                    "contract_id": contract_id,
                    "verdict":     findings.get("verdict", "")
                },
                ids=[f"{contract_id}_{clause_type}"]
            )
            stored_any = True
            _log(f"_store_clauses: stored {clause_type} for {vendor_name}")

    if stored_any:
        # Reset singleton to force index flush
        global _client
        _client = None
        _log("_store_clauses: client reset to force index flush")


# ── Vendor name fuzzy matching ────────────────────────────────────────────────

def _vendor_words(name):
    skip = {"the", "and", "for", "pte", "ltd", "llc", "inc",
            "corp", "co", "plc", "gmbh", "bv", "ag", "sas"}
    return set(
        w.lower() for w in name.replace(".", "").replace(",", "").split()
        if len(w) > 3 and w.lower() not in skip
    )


def _vendor_matches(query_name, stored_name):
    query_words  = _vendor_words(query_name)
    stored_words = _vendor_words(stored_name)
    if not query_words or not stored_words:
        return False
    return bool(query_words & stored_words)


# ── Query ─────────────────────────────────────────────────────────────────────

def query_vendor_history(vendor_name, n=5):
    if not CHROMA_AVAILABLE:
        return []
    try:
        col   = _col(COLLECTION_CONTRACTS)
        total = col.count()
        _log(f"query_vendor_history: vendor='{vendor_name}' total_docs={total}")

        if total == 0:
            _log("query_vendor_history: collection empty — returning []")
            return []

        safe_n = min(n, total)
        _log(f"query_vendor_history: safe_n={safe_n}")

        # Step 1: exact match
        try:
            results   = col.query(
                query_texts=[f"vendor {vendor_name}"],
                n_results=safe_n,
                where={"vendor": vendor_name}
            )
            formatted = _fmt(results)
            _log(f"query_vendor_history: exact match returned "
                 f"{len(formatted)} result(s)")
            if formatted:
                return formatted
        except Exception as e:
            _log(f"query_vendor_history: exact match EXCEPTION — {e}")

        # Step 2: semantic search without filter, fuzzy match in Python
        safe_n2 = min(n * 3, total)
        _log(f"query_vendor_history: semantic fallback n_results={safe_n2}")
        try:
            results   = col.query(
                query_texts=[f"vendor contract {vendor_name}"],
                n_results=safe_n2
            )
            formatted = _fmt(results)
            _log(f"query_vendor_history: semantic fallback returned "
                 f"{len(formatted)} result(s)")

            if not formatted:
                _log("query_vendor_history: semantic fallback empty — returning []")
                return []

            matches = [
                r for r in formatted
                if _vendor_matches(
                    vendor_name,
                    r.get("metadata", {}).get("vendor", "")
                )
            ]
            _log(f"query_vendor_history: fuzzy match found {len(matches)} "
                 f"match(es) from vendors: "
                 f"{[r.get('metadata',{}).get('vendor','') for r in formatted]}")
            return matches[:n]

        except Exception as e:
            _log(f"query_vendor_history: semantic fallback EXCEPTION — {e}")
            return []

    except Exception as e:
        _log(f"query_vendor_history: outer EXCEPTION — {e}")
        return []


def query_similar_clauses(clause_text, clause_type, n=5):
    if not CHROMA_AVAILABLE:
        return []
    try:
        col   = _col(COLLECTION_CLAUSES)
        total = col.count()
        _log(f"query_similar_clauses: type='{clause_type}' total_docs={total}")

        if total == 0:
            _log("query_similar_clauses: collection empty — returning []")
            return []

        safe_n = min(n, total)

        # Step 1: exact type match
        try:
            results   = col.query(
                query_texts=[clause_text],
                n_results=safe_n,
                where={"type": clause_type}
            )
            formatted = _fmt(results)
            _log(f"query_similar_clauses: exact match returned "
                 f"{len(formatted)} result(s)")
            if formatted:
                return formatted
        except Exception as e:
            _log(f"query_similar_clauses: exact match EXCEPTION — {e}")

        # Step 2: semantic search without filter
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
        return matches[:n]

    except Exception as e:
        _log(f"query_similar_clauses: EXCEPTION — {e}")
        return []


def query_by_topic(topic, n=5):
    if not CHROMA_AVAILABLE:
        return []
    try:
        col   = _col(COLLECTION_CONTRACTS)
        total = col.count()
        _log(f"query_by_topic: topic='{topic[:50]}' total_docs={total}")

        if total == 0:
            return []

        safe_n  = min(n, total)
        results = col.query(
            query_texts=[topic],
            n_results=safe_n
        )
        formatted = _fmt(results)
        _log(f"query_by_topic: returned {len(formatted)} result(s)")
        return formatted

    except Exception as e:
        _log(f"query_by_topic: EXCEPTION — {e}")
        return []


def get_all_contracts(n=100):
    if not CHROMA_AVAILABLE:
        return []
    try:
        results = _col(COLLECTION_CONTRACTS).get(limit=n)
        if not results or not results.get("documents"):
            return []
        return [
            {
                "document": doc,
                "metadata": results["metadatas"][i]
                            if results.get("metadatas") else {}
            }
            for i, doc in enumerate(results["documents"])
        ]
    except Exception:
        return []


def get_stats():
    stats = {"contracts": 0, "clauses": 0, "vendors": set()}
    if not CHROMA_AVAILABLE:
        stats["vendors"] = []
        return stats
    try:
        data = _col(COLLECTION_CONTRACTS).get()
        if data and data.get("metadatas"):
            stats["contracts"] = len(data["metadatas"])
            for m in data["metadatas"]:
                if m.get("vendor"):
                    stats["vendors"].add(m["vendor"])
    except Exception:
        pass
    try:
        data = _col(COLLECTION_CLAUSES).get()
        if data and data.get("ids"):
            stats["clauses"] = len(data["ids"])
    except Exception:
        pass
    stats["vendors"] = list(stats["vendors"])
    return stats


def is_available():
    return CHROMA_AVAILABLE


def _fmt(results):
    out = []
    if not results or not results.get("documents"):
        return out
    docs  = results["documents"][0] if results.get("documents") else []
    metas = results["metadatas"][0] if results.get("metadatas") else []
    dists = results["distances"][0] if results.get("distances") else []
    for i, doc in enumerate(docs):
        out.append({
            "document":   doc,
            "metadata":   metas[i] if i < len(metas) else {},
            "similarity": round(1 - dists[i], 3) if i < len(dists) else None
        })
    return out