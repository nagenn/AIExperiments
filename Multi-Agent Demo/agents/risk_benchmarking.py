"""
agents/risk_benchmarking.py
----------------------------
AGENT 2 — RISK & BENCHMARKING AGENT

Tools:
  • read_contract         — read the contract independently
  • benchmark_clause      — compare against industry standards
  • request_clarification — ask Agent 1 to re-extract a clause
  • flag_risk             — record a risk finding
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (get_client, read_contract as util_read_contract,
                   prompt_contract_file, header, section, log,
                   sending, receiving, risk_line,
                   unwrap_result, run_tool_loop)
from message_bus import send_message, get_messages, mark_read, wait_for_message

AGENT_NAME = "risk_benchmarking"
client     = get_client()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_contract",
            "description": "Read and parse the contract file.",
            "parameters": {
                "type": "object",
                "properties": {"filepath": {"type": "string"}},
                "required": ["filepath"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "benchmark_clause",
            "description": "Benchmark a clause against industry standards.",
            "parameters": {
                "type": "object",
                "properties": {
                    "clause_type": {
                        "type": "string",
                        "enum": ["liability_cap", "payment_terms", "sla",
                                 "termination_notice", "data_breach_notification",
                                 "amendment_notice"]
                    },
                    "current_value": {"type": "string"}
                },
                "required": ["clause_type", "current_value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "request_clarification",
            "description": "Ask Agent 1 to re-extract an ambiguous clause.",
            "parameters": {
                "type": "object",
                "properties": {
                    "clause": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["clause", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flag_risk",
            "description": "Record a risk finding with rating and evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "area":      {"type": "string"},
                    "level":     {"type": "string",
                                  "enum": ["CRITICAL", "HIGH", "MEDIUM", "LOW"]},
                    "finding":   {"type": "string"},
                    "benchmark": {"type": "string"}
                },
                "required": ["area", "level", "finding"]
            }
        }
    }
]

SYSTEM_PROMPT = """
You are the Risk & Benchmarking Agent — a senior commercial risk specialist.

Use tools in this order:
1. read_contract (skip if extraction provided)
2. benchmark_clause — for liability, payment, SLA
3. request_clarification — if a clause is ambiguous
4. flag_risk — for each risk area (call multiple times)

BENCHMARKS:
- Liability cap: standard 6-12 months. Below 3 months = CRITICAL.
- Payment: standard Net 30. 60+ days = HIGH.
- SLA: standard 99.9%. Below 99.5% = HIGH. Below 99.0% = CRITICAL.
- Termination: standard 30-90 days. 180+ = HIGH.
- Amendment notice: minimum 30 days. Below 14 days = CRITICAL.

IMPORTANT: Return your final answer as a raw JSON object only.
No preamble, no explanation, no markdown fences. Just the JSON.

Final answer format:
{
  "overall_risk": "RED|AMBER|GREEN",
  "liability_risk": "", "liability_finding": "", "liability_benchmark": "",
  "payment_risk": "", "payment_finding": "", "payment_benchmark": "",
  "data_risk": "", "data_finding": "",
  "ip_risk": "", "ip_finding": "",
  "termination_risk": "", "termination_finding": "",
  "sla_risk": "", "sla_finding": "", "sla_benchmark": "",
  "amendment_risk": "", "amendment_finding": "",
  "breach_notification_risk": "", "breach_notification_finding": "",
  "top_3_risks": []
}
"""

BENCHMARKS = {
    "liability_cap":            {"standard": "6-12 months fees",   "critical": "below 1 month"},
    "payment_terms":            {"standard": "Net 30 days",         "critical": "60+ days"},
    "sla":                      {"standard": "99.9%",               "critical": "below 99.0%"},
    "termination_notice":       {"standard": "30-90 days",          "critical": "180+ days"},
    "data_breach_notification": {"standard": "24-72 hours",         "critical": "30+ days"},
    "amendment_notice":         {"standard": "30 days minimum",     "critical": "below 14 days"},
}

_state = {"contract_text": "", "risks": [], "clarifications": {}}


def handle_tool(name, args):
    if name == "read_contract":
        try:
            text = util_read_contract(args.get("filepath", ""))
            _state["contract_text"] = text
            return "Contract read."
        except Exception as e:
            return f"Error: {e}"

    elif name == "benchmark_clause":
        ctype   = args.get("clause_type", "")
        current = args.get("current_value", "")
        bench   = BENCHMARKS.get(ctype, {})
        if bench:
            return (f"{ctype}: current='{current}'. "
                    f"Standard: {bench.get('standard','')}. "
                    f"Critical threshold: {bench.get('critical','')}.")
        return f"No benchmark for {ctype}."

    elif name == "request_clarification":
        clause = args.get("clause", "")
        reason = args.get("reason", "")
        _state["clarifications"][clause] = reason
        sending("contract_intelligence", f"Need clarification: {clause}")
        return f"Clarification requested: {clause}."

    elif name == "flag_risk":
        _state["risks"].append(args)
        return f"Flagged: {args.get('area','')} = {args.get('level','')}."

    return f"Unknown tool: {name}"


def display_output(result):
    result = unwrap_result(result)
    section("RISK & BENCHMARKING — ASSESSMENT")

    overall = result.get("overall_risk", "?")
    icon    = {"RED": "🔴", "AMBER": "🟡", "GREEN": "🟢"}.get(overall, "⚪")
    print(f"\n  {icon}  OVERALL: {overall}\n")

    areas = [
        ("Liability",           "liability_risk",            "liability_finding",           "liability_benchmark"),
        ("Payment",             "payment_risk",              "payment_finding",             "payment_benchmark"),
        ("Data rights",         "data_risk",                 "data_finding",                None),
        ("IP",                  "ip_risk",                   "ip_finding",                  None),
        ("Termination",         "termination_risk",          "termination_finding",         None),
        ("SLA",                 "sla_risk",                  "sla_finding",                 "sla_benchmark"),
        ("Amendments",          "amendment_risk",            "amendment_finding",           None),
        ("Breach notification", "breach_notification_risk",  "breach_notification_finding", None),
    ]
    for label, rk, fk, bk in areas:
        level   = result.get(rk, "")
        finding = result.get(fk, "")
        bench   = result.get(bk, "") if bk else ""
        if level:
            risk_line(label, level, finding)
            if bench:
                print(f"     📊 {bench}")

    if result.get("top_3_risks"):
        section("TOP 3 RISKS")
        for i, r in enumerate(result["top_3_risks"], 1):
            log(f"  {i}. {r}")


def summary_for_controller(result):
    result  = unwrap_result(result)
    overall = result.get("overall_risk", "?")
    icon    = {"RED": "🔴", "AMBER": "🟡", "GREEN": "🟢"}.get(overall, "⚪")
    lines   = [f"  {icon}  Overall risk: {overall}"]
    for r in result.get("top_3_risks", [])[:3]:
        lines.append(f"  ⚠️   {r[:90]}")
    return "\n".join(lines)


def run_as_service():
    header("AGENT 2 — RISK & BENCHMARKING",
           "Tools: read_contract | benchmark_clause | "
           "request_clarification | flag_risk")
    log("Waiting for contracts...\n")

    while True:
        for msg_type in ("standalone_request", "collab_request"):
            msgs = get_messages(AGENT_NAME, message_type=msg_type)
            for filepath, msg in msgs:
                mark_read(filepath)
                contract_file = msg["content"]["contract_file"]
                contract_text = msg["content"]["contract_text"]
                extraction    = unwrap_result(msg["content"].get("extraction", {}))
                is_collab     = (msg_type == "collab_request")

                _state["contract_text"]  = contract_text
                _state["risks"]          = []
                _state["clarifications"] = {}

                header("AGENT 2 — RISK & BENCHMARKING",
                       f"{'Collab' if is_collab else 'Standalone'}: "
                       f"{os.path.basename(contract_file)}")

                context = (
                    f"Review: {contract_file}\n\n"
                    f"{'Extraction from Agent 1:' + json.dumps(extraction) if extraction else 'Contract:'}\n"
                    f"{'' if extraction else contract_text[:8000]}"
                )

                result = run_tool_loop(
                    client, SYSTEM_PROMPT, context, TOOLS, handle_tool
                )

                # Handle clarification requests
                if _state["clarifications"] and is_collab:
                    for clause, reason in _state["clarifications"].items():
                        send_message(AGENT_NAME, "contract_intelligence",
                                     "reextract_request", {
                                         "clause":        clause,
                                         "contract_text": contract_text,
                                         "reason":        reason
                                     })
                        log("  Waiting for Agent 1 clarification...")
                        try:
                            resp = wait_for_message(AGENT_NAME,
                                                    "reextract_response",
                                                    timeout=120)
                            clarified = unwrap_result(
                                resp["content"].get("result", {}))
                            receiving("contract_intelligence",
                                      f"Clarification received for: {clause}")
                            if extraction:
                                extraction.update(
                                    {k: v for k, v in clarified.items() if v})
                        except TimeoutError:
                            log("  No response — proceeding.")

                display_output(result)

                if is_collab:
                    sending("orchestrator", "Risk assessment complete")
                    send_message(AGENT_NAME, "orchestrator", "risk_complete", {
                        "contract_file": contract_file,
                        "contract_text": contract_text,
                        "extraction":    extraction,
                        "risk_findings": result
                    })
                else:
                    send_message(AGENT_NAME, "controller", "standalone_result", {
                        "agent":       AGENT_NAME,
                        "summary":     summary_for_controller(result),
                        "full_result": result
                    })
                log("\n  ✅ Done. Waiting...\n")

        # Policy challenges
        msgs = get_messages(AGENT_NAME, message_type="policy_challenge")
        for filepath, msg in msgs:
            mark_read(filepath)
            from_agent = msg["from"]
            challenge  = msg["content"]

            header("AGENT 2 — RISK & BENCHMARKING",
                   "Policy challenge from Agent 3")
            receiving("policy_compliance",
                      f"{challenge.get('rule')}: {challenge.get('challenge','')[:80]}")

            response = run_tool_loop(
                client, SYSTEM_PROMPT,
                f"""Policy challenge received:
{json.dumps(challenge, indent=2)}
Defend your rating or concede.
Return JSON only: {{"decision":"defend|upgrade","new_rating":"","response":""}}""",
                TOOLS, handle_tool
            )
            response = unwrap_result(response)
            decision = response.get("decision", "defend")

            if decision == "upgrade":
                log("  ✅ Conceding — upgrading rating.")
            else:
                log("  🛡️  Defending rating.")

            send_message(AGENT_NAME, from_agent, "challenge_response", {
                "rule":     challenge.get("rule"),
                "decision": decision,
                "response": response
            })

        time.sleep(1)


if __name__ == "__main__":
    if "--service" in sys.argv:
        run_as_service()
    else:
        contract_file = (sys.argv[1] if len(sys.argv) > 1
                         else prompt_contract_file())
        header("AGENT 2 — RISK & BENCHMARKING",
               f"Standalone: {os.path.basename(contract_file)}")
        log("Tools: read_contract | benchmark_clause | "
            "request_clarification | flag_risk\n")
        contract_text           = util_read_contract(contract_file)
        _state["contract_text"] = contract_text
        result = run_tool_loop(
            client, SYSTEM_PROMPT,
            f"Review: {contract_file}\n\nContract:\n{contract_text[:8000]}",
            TOOLS, handle_tool
        )
        display_output(result)
