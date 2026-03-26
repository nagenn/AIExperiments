"""
agents/contract_intelligence.py
--------------------------------
AGENT 1 — CONTRACT INTELLIGENCE AGENT

Tools:
  • read_contract          — parse the contract file
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
                   sending, receiving, unwrap_result, run_tool_loop)
from message_bus import send_message, get_messages, mark_read

AGENT_NAME = "contract_intelligence"
client     = get_client()

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
                    "main_concerns":  {"type": "string"}
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
2. identify_clauses
3. flag_ambiguity (for each ambiguous clause)
4. summarise_plain_english

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
    if result.get("ambiguous_clauses"):
        lines.append(f"  ⚠️  {len(result['ambiguous_clauses'])} ambiguous clause(s)")
    lines.append(f"\n  {result.get('plain_english_summary','')[:200]}")
    return "\n".join(lines)


def run_as_service():
    header("AGENT 1 — CONTRACT INTELLIGENCE",
           "Tools: read_contract | identify_clauses | "
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
        log("Tools: read_contract | identify_clauses | "
            "flag_ambiguity | summarise_plain_english\n")
        contract_text           = util_read_contract(contract_file)
        _state["contract_text"] = contract_text
        result = run_tool_loop(
            client, SYSTEM_PROMPT,
            f"Review: {contract_file}\n\nContract:\n{contract_text[:8000]}",
            TOOLS, handle_tool
        )
        display_output(result, contract_file)
