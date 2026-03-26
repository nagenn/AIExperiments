"""
agents/contract_intelligence.py
--------------------------------
AGENT 1 — CONTRACT INTELLIGENCE AGENT

Tools:
  • read_contract          — parse the contract file
  • query_memory           — search organisational memory (via MCP server)
  • identify_clauses       — extract and structure all key clauses
  • flag_ambiguity         — flag a clause as ambiguous
  • summarise_plain_english — write a plain English summary
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (get_client, read_contract as util_read_contract,
                   prompt_contract_file, header, section, log,
                   memory_hit, sending, receiving, unwrap_result,
                   run_tool_loop)
from message_bus import send_message, get_messages, mark_read

AGENT_NAME = "contract_intelligence"
client     = get_client()


# ── MCP memory client ─────────────────────────────────────────────────────────

MCP_URL = "http://127.0.0.1:8765/sse"

def _call_mcp(tool_name, tool_args):
    """Call the Chroma MCP server over SSE/HTTP."""
    import asyncio
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async def _run():
        async with sse_client(url=MCP_URL) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, tool_args)
                return result.content[0].text

    return asyncio.run(_run())


# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_contract",
            "description": "Read and parse the contract file. Always call this first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"}
                },
                "required": ["filepath"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_memory",
            "description": (
                "Search organisational memory for prior contracts, vendor history, "
                "or similar clauses. Call after reading the contract."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["vendor_history", "similar_clauses", "topic"]
                    },
                    "vendor_name": {"type": "string"},
                    "clause_text": {"type": "string"},
                    "clause_type": {
                        "type": "string",
                        "enum": ["liability_cap", "payment_terms", "data_rights", "sla"]
                    },
                    "topic": {"type": "string"}
                },
                "required": ["query_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "identify_clauses",
            "description": "Extract and structure all key clauses from the contract.",
            "parameters": {
                "type": "object",
                "properties": {
                    "contract_text": {"type": "string"}
                },
                "required": ["contract_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flag_ambiguity",
            "description": "Flag a specific clause as ambiguous or unclear.",
            "parameters": {
                "type": "object",
                "properties": {
                    "clause":  {"type": "string"},
                    "reason":  {"type": "string"},
                    "impact":  {"type": "string"}
                },
                "required": ["clause", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarise_plain_english",
            "description": "Write a plain English summary for a non-lawyer business user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key_facts":      {"type": "string"},
                    "main_concerns":  {"type": "string"},
                    "memory_context": {"type": "string"}
                },
                "required": ["key_facts", "main_concerns"]
            }
        }
    }
]

SYSTEM_PROMPT = """
You are the Contract Intelligence Agent — a specialist in reading and
structuring contracts for business users.

Use tools in this order:
1. read_contract
2. query_memory (vendor_history)
3. identify_clauses
4. flag_ambiguity (for each ambiguous clause)
5. summarise_plain_english

Always call query_memory. If memory returns results, incorporate them.

IMPORTANT: Return your final answer as a raw JSON object only.
No preamble, no explanation, no markdown fences. Just the JSON.

Final answer format:
{
  "vendor_name": "",
  "contract_type": "",
  "contract_duration": "",
  "payment_amount": "",
  "payment_due_days": "",
  "late_payment_penalty": "",
  "liability_cap": "",
  "liability_cap_amount": "",
  "data_ownership": "",
  "data_usage_rights": "",
  "ip_outputs_owner": "",
  "termination_by_customer": "",
  "notice_period_days": "",
  "data_return_on_exit": "",
  "governing_law": "",
  "uptime_sla": "",
  "amendment_rights": "",
  "ambiguous_clauses": [],
  "unusual_clauses": [],
  "memory_findings": "",
  "plain_english_summary": ""
}
"""

_state = {"contract_text": "", "flags": []}


def handle_tool(name, args):
    if name == "read_contract":
        try:
            text = util_read_contract(args.get("filepath", ""))
            _state["contract_text"] = text
            return "Contract read."
        except Exception as e:
            return f"Error: {e}"

    elif name == "query_memory":
        qtype = args.get("query_type", "topic")
        try:
            if qtype == "vendor_history":
                vendor  = args.get("vendor_name", "")
                results = json.loads(_call_mcp("query_vendor_history",
                                               {"vendor_name": vendor}))
                if not results:
                    return f"No prior contracts with '{vendor}'. First encounter."
                lines = [f"Found {len(results)} prior contract(s) with {vendor}:"]
                for r in results[:3]:
                    m = r.get("metadata", {})
                    lines.append(
                        f"  • {m.get('review_date','?')[:10]} — "
                        f"Verdict: {m.get('verdict','?')} — "
                        f"Risk: {m.get('overall_risk','?')} — "
                        f"Liability: {m.get('liability_cap','?')}"
                    )
                memory_hit(" | ".join(lines[:2]))
                return "\n".join(lines)

            elif qtype == "similar_clauses":
                text    = args.get("clause_text", "")
                ctype   = args.get("clause_type", "liability_cap")
                results = json.loads(_call_mcp("query_similar_clauses",
                                               {"clause_text": text,
                                                "clause_type": ctype}))
                if not results:
                    return f"No similar {ctype} clauses in memory."
                lines = [f"Found {len(results)} similar {ctype} clause(s):"]
                for r in results[:3]:
                    m = r.get("metadata", {})
                    lines.append(f"  • {m.get('vendor','?')}: {r['document'][:80]}")
                memory_hit(lines[0])
                return "\n".join(lines)

            elif qtype == "topic":
                topic   = args.get("topic", "")
                results = json.loads(_call_mcp("query_vendor_history",
                                               {"vendor_name": topic}))
                if not results:
                    return "No memory found for this topic."
                lines = [f"Found {len(results)} relevant item(s):"]
                for r in results[:2]:
                    lines.append(f"  • {r['document'][:100]}")
                memory_hit(lines[0])
                return "\n".join(lines)

        except Exception as e:
            return f"Memory unavailable: {e}"

        return "Unknown query type."

    elif name == "identify_clauses":
        return "Clause identification complete."

    elif name == "flag_ambiguity":
        _state["flags"].append(args.get("clause", ""))
        return f"Flagged: {args.get('clause','')} — {args.get('reason','')}"

    elif name == "summarise_plain_english":
        return "Plain English summary drafted."

    return f"Unknown tool: {name}"


def display_output(result, contract_file):
    result = unwrap_result(result)
    section("CONTRACT INTELLIGENCE — FINDINGS")
    fields = [
        ("Vendor",            "vendor_name"),
        ("Type",              "contract_type"),
        ("Duration",          "contract_duration"),
        ("Payment",           "payment_amount"),
        ("Payment due",       "payment_due_days"),
        ("Late penalty",      "late_payment_penalty"),
        ("Liability cap",     "liability_cap_amount"),
        ("Data ownership",    "data_ownership"),
        ("Data usage",        "data_usage_rights"),
        ("IP outputs",        "ip_outputs_owner"),
        ("Exit notice",       "notice_period_days"),
        ("Data on exit",      "data_return_on_exit"),
        ("Governing law",     "governing_law"),
        ("SLA",               "uptime_sla"),
        ("Amendments",        "amendment_rights"),
    ]
    for label, field in fields:
        v = result.get(field, "")
        if v:
            print(f"  • {label:18s}: {v}")

    if result.get("memory_findings"):
        print(f"\n  🧠 {result['memory_findings']}")

    if result.get("ambiguous_clauses"):
        print(f"\n  ⚠️  Ambiguous clauses:")
        for c in result["ambiguous_clauses"]:
            print(f"     • {c}")

    if result.get("plain_english_summary"):
        section("PLAIN ENGLISH SUMMARY")
        log(result["plain_english_summary"])


def summary_for_controller(result):
    result = unwrap_result(result)
    lines = [
        f"  • Vendor        : {result.get('vendor_name','?')}",
        f"  • Type          : {result.get('contract_type','?')}",
        f"  • Duration      : {result.get('contract_duration','?')}",
        f"  • Payment       : {result.get('payment_amount','?')} "
                             f"(due {result.get('payment_due_days','?')} days)",
        f"  • Liability cap : {result.get('liability_cap_amount','?')}",
        f"  • SLA           : {result.get('uptime_sla','?')}",
        f"  • Governing law : {result.get('governing_law','?')}",
    ]
    if result.get("memory_findings"):
        lines.append(f"\n  🧠 {result['memory_findings'][:120]}")
    if result.get("ambiguous_clauses"):
        lines.append(f"  ⚠️  {len(result['ambiguous_clauses'])} ambiguous clause(s)")
    lines.append(f"\n  {result.get('plain_english_summary','')[:200]}")
    return "\n".join(lines)


def run_as_service():
    header("AGENT 1 — CONTRACT INTELLIGENCE",
           "Tools: read_contract | query_memory | identify_clauses | "
           "flag_ambiguity | summarise_plain_english")
    log("Waiting for contracts...\n")

    while True:
        for msg_type in ("standalone_request", "collab_request"):
            msgs = get_messages(AGENT_NAME, message_type=msg_type)
            for filepath, msg in msgs:
                mark_read(filepath)
                contract_file = msg["content"]["contract_file"]
                contract_text = msg["content"]["contract_text"]
                is_collab     = (msg_type == "collab_request")

                _state["contract_text"] = contract_text
                _state["flags"]         = []

                header("AGENT 1 — CONTRACT INTELLIGENCE",
                       f"{'Collab' if is_collab else 'Standalone'}: "
                       f"{os.path.basename(contract_file)}")

                result = run_tool_loop(
                    client, SYSTEM_PROMPT,
                    f"Review: {contract_file}\n\nContract:\n{contract_text[:8000]}",
                    TOOLS, handle_tool
                )

                display_output(result, contract_file)

                if is_collab:
                    sending("orchestrator", "Extraction complete")
                    send_message(AGENT_NAME, "orchestrator",
                                 "extraction_complete", {
                                     "contract_file": contract_file,
                                     "contract_text": contract_text,
                                     "extraction":    result
                                 })
                else:
                    send_message(AGENT_NAME, "controller",
                                 "standalone_result", {
                                     "agent":       AGENT_NAME,
                                     "summary":     summary_for_controller(result),
                                     "full_result": result
                                 })
                log("\n  ✅ Done. Waiting...\n")

        # Re-extraction requests from peers
        msgs = get_messages(AGENT_NAME, message_type="reextract_request")
        for filepath, msg in msgs:
            mark_read(filepath)
            from_agent    = msg["from"]
            clause        = msg["content"]["clause"]
            contract_text = msg["content"].get("contract_text",
                                               _state.get("contract_text", ""))

            header("AGENT 1 — CONTRACT INTELLIGENCE",
                   f"Re-extraction request from {from_agent.upper()}")
            receiving(from_agent, f"Re-extract: {clause}")

            result = run_tool_loop(
                client, SYSTEM_PROMPT,
                f"Re-extract only: {clause}\n\nContract:\n{contract_text[:8000]}",
                TOOLS, handle_tool
            )

            send_message(AGENT_NAME, from_agent, "reextract_response", {
                "clause": clause,
                "result": result
            })

        time.sleep(1)


if __name__ == "__main__":
    if "--service" in sys.argv:
        run_as_service()
    else:
        contract_file = (sys.argv[1] if len(sys.argv) > 1
                         else prompt_contract_file())
        header("AGENT 1 — CONTRACT INTELLIGENCE",
               f"Standalone: {os.path.basename(contract_file)}")
        log("Tools: read_contract | query_memory | identify_clauses | "
            "flag_ambiguity | summarise_plain_english\n")
        contract_text           = util_read_contract(contract_file)
        _state["contract_text"] = contract_text
        result = run_tool_loop(
            client, SYSTEM_PROMPT,
            f"Review: {contract_file}\n\nContract:\n{contract_text[:8000]}",
            TOOLS, handle_tool
        )
        display_output(result, contract_file)
