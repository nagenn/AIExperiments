"""
agents/negotiation_advisor.py
------------------------------
AGENT 4 — NEGOTIATION ADVISOR AGENT

Tools:
  • read_contract            — read the contract independently
  • draft_redline            — draft a specific clause rewrite
  • build_strategy           — build negotiation strategy
  • issue_negotiation_brief  — issue the final brief
"""

import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (get_client, read_contract as util_read_contract,
                   prompt_contract_file, header, section, log,
                   sending, unwrap_result, run_tool_loop)
from message_bus import send_message, get_messages, mark_read

AGENT_NAME = "negotiation_advisor"
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
            "name": "draft_redline",
            "description": "Draft a specific clause rewrite — actual replacement wording.",
            "parameters": {
                "type": "object",
                "properties": {
                    "clause_reference": {"type": "string"},
                    "issue":            {"type": "string"},
                    "current_wording":  {"type": "string"},
                    "proposed_wording": {"type": "string"},
                    "rationale":        {"type": "string"},
                    "priority":         {"type": "string",
                                         "enum": ["CRITICAL", "HIGH", "MEDIUM"]}
                },
                "required": ["clause_reference", "issue",
                             "current_wording", "proposed_wording", "priority"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "build_strategy",
            "description": "Build the negotiation strategy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dealbreakers":      {"type": "array", "items": {"type": "string"}},
                    "nice_to_haves":     {"type": "array", "items": {"type": "string"}},
                    "opening_position":  {"type": "string"},
                    "fallback_position": {"type": "string"},
                    "leverage_points":   {"type": "array", "items": {"type": "string"}}
                },
                "required": ["dealbreakers", "opening_position"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "issue_negotiation_brief",
            "description": "Issue the final negotiation brief.",
            "parameters": {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": ["NEGOTIATE", "WALK_AWAY", "ACCEPT_WITH_CONDITIONS"]
                    },
                    "summary":               {"type": "string"},
                    "walk_away_conditions":  {"type": "array", "items": {"type": "string"}}
                },
                "required": ["verdict", "summary"]
            }
        }
    }
]

SYSTEM_PROMPT = """
You are the Negotiation Advisor Agent.

Use tools in this order:
1. read_contract (skip if context provided)
2. draft_redline — for each clause that needs changing (call multiple times)
3. build_strategy — overall negotiation approach
4. issue_negotiation_brief — final brief

REDLINE PRINCIPLES:
- Liability: propose minimum 12 months fees
- Payment: propose Net 30
- Data rights: propose complete removal of vendor data licence
- IP: propose customer owns all outputs
- Termination: propose 60 days, no post-termination liability
- SLA: propose 99.9% with financial credits
- Amendment: propose mutual written consent

IMPORTANT: Return your final answer as a raw JSON object only.
No preamble, no explanation, no markdown fences. Just the JSON.

Final answer format:
{
  "negotiation_verdict": "NEGOTIATE|WALK_AWAY|ACCEPT_WITH_CONDITIONS",
  "dealbreakers": [],
  "nice_to_haves": [],
  "redline_suggestions": [
    {
      "clause_reference": "",
      "issue": "",
      "current_wording": "",
      "proposed_wording": "",
      "rationale": "",
      "priority": "CRITICAL|HIGH|MEDIUM"
    }
  ],
  "negotiation_strategy": "",
  "opening_position": "",
  "fallback_position": "",
  "leverage_points": [],
  "walk_away_conditions": []
}
"""

_state = {"contract_text": "", "redlines": []}


def handle_tool(name, args):
    if name == "read_contract":
        try:
            text = util_read_contract(args.get("filepath", ""))
            _state["contract_text"] = text
            return "Contract read."
        except Exception as e:
            return f"Error: {e}"

    elif name == "draft_redline":
        redline = {k: args.get(k, "") for k in [
            "clause_reference", "issue", "current_wording",
            "proposed_wording", "rationale", "priority"
        ]}
        _state["redlines"].append(redline)
        return f"Redline: {redline['clause_reference']} [{redline['priority']}]"

    elif name == "build_strategy":
        return (f"Strategy built. Dealbreakers: {len(args.get('dealbreakers',[]))}.")

    elif name == "issue_negotiation_brief":
        verdict = args.get("verdict", "")
        icons   = {"NEGOTIATE": "🟡", "WALK_AWAY": "🔴",
                   "ACCEPT_WITH_CONDITIONS": "🟢"}
        log(f"\n  {icons.get(verdict,'⚪')}  VERDICT: {verdict.replace('_',' ')}")
        log(f"  {args.get('summary','')[:120]}")
        return f"Brief issued: {verdict}."

    return f"Unknown tool: {name}"


def display_output(result):
    result = unwrap_result(result)
    section("NEGOTIATION ADVISOR — BRIEF")

    nv    = result.get("negotiation_verdict", "?")
    icons = {"NEGOTIATE": "🟡", "WALK_AWAY": "🔴",
             "ACCEPT_WITH_CONDITIONS": "🟢"}
    icon  = icons.get(nv, "⚪")
    print(f"\n  {icon}  VERDICT: {nv.replace('_',' ')}\n")

    if result.get("dealbreakers"):
        print("  🔴 Dealbreakers:")
        for d in result["dealbreakers"]:
            print(f"     • {d}")

    if result.get("nice_to_haves"):
        print("\n  🟡 Nice-to-haves:")
        for n in result["nice_to_haves"]:
            print(f"     • {n}")

    redlines = result.get("redline_suggestions", [])
    if redlines:
        section(f"REDLINES ({len(redlines)} clauses)")
        for r in redlines:
            p    = r.get("priority", "")
            icon = {"CRITICAL": "🔴", "HIGH": "🟡", "MEDIUM": "⚪"}.get(p, "")
            print(f"\n  {icon} {r.get('clause_reference','')} — {r.get('issue','')}")
            print(f"     Current : {r.get('current_wording','')[:100]}")
            print(f"     Proposed: {r.get('proposed_wording','')[:100]}")

    if result.get("negotiation_strategy"):
        section("STRATEGY")
        log(result["negotiation_strategy"][:300])

    if result.get("opening_position"):
        log(f"\n  Opening : {result['opening_position'][:120]}")
    if result.get("fallback_position"):
        log(f"  Fallback: {result['fallback_position'][:120]}")

    if result.get("walk_away_conditions"):
        section("🚪 WALK AWAY IF")
        for w in result["walk_away_conditions"]:
            log(f"  • {w}")


def summary_for_controller(result):
    result  = unwrap_result(result)
    nv      = result.get("negotiation_verdict", "?")
    icons   = {"NEGOTIATE": "🟡", "WALK_AWAY": "🔴",
               "ACCEPT_WITH_CONDITIONS": "🟢"}
    icon    = icons.get(nv, "⚪")
    lines   = [f"  {icon}  Verdict: {nv.replace('_',' ')}"]
    for d in result.get("dealbreakers", [])[:3]:
        lines.append(f"  🔴  {d[:90]}")
    redlines = result.get("redline_suggestions", [])
    if redlines:
        lines.append(f"  ✏️   Redlines drafted: {len(redlines)} clauses")
    return "\n".join(lines)


def save_brief(result, vendor, contract_file):
    result = unwrap_result(result)
    os.makedirs("negotiation_briefs", exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    vendor_s = vendor.replace(" ", "_").replace(".", "")[:30]
    filename = f"negotiation_briefs/{vendor_s}_{ts}.txt"
    with open(filename, "w") as f:
        f.write(f"NEGOTIATION BRIEF\n"
                f"Contract: {contract_file}\n"
                f"Vendor: {vendor}\n"
                f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}\n"
                f"{'='*60}\n\n")
        f.write(f"VERDICT: {result.get('negotiation_verdict','?')}\n\n")
        for r in result.get("redline_suggestions", []):
            f.write(f"\n[{r.get('priority','')}] {r.get('clause_reference','')}\n")
            f.write(f"Issue:    {r.get('issue','')}\n")
            f.write(f"Current:  {r.get('current_wording','')}\n")
            f.write(f"Proposed: {r.get('proposed_wording','')}\n")
        f.write(f"\nSTRATEGY:\n{result.get('negotiation_strategy','')}\n")
        f.write(f"\nOPENING:  {result.get('opening_position','')}\n")
        f.write(f"FALLBACK: {result.get('fallback_position','')}\n")
    log(f"  Brief saved: {filename}")


def run_as_service():
    header("AGENT 4 — NEGOTIATION ADVISOR",
           "Tools: read_contract | draft_redline | "
           "build_strategy | issue_negotiation_brief")
    log("Waiting for contracts...\n")

    while True:
        for msg_type in ("standalone_request", "collab_request"):
            msgs = get_messages(AGENT_NAME, message_type=msg_type)
            for filepath, msg in msgs:
                mark_read(filepath)
                contract_file   = msg["content"]["contract_file"]
                contract_text   = msg["content"]["contract_text"]
                extraction      = unwrap_result(msg["content"].get("extraction", {}))
                risk_findings   = unwrap_result(msg["content"].get("risk_findings", {}))
                policy_findings = unwrap_result(msg["content"].get("policy_findings", {}))
                is_collab       = (msg_type == "collab_request")

                _state["contract_text"] = contract_text
                _state["redlines"]      = []

                header("AGENT 4 — NEGOTIATION ADVISOR",
                       f"{'Collab' if is_collab else 'Standalone'}: "
                       f"{os.path.basename(contract_file)}")

                context = (
                    f"Negotiation brief for: {contract_file}\n\n"
                    + (f"Agent 1 extraction:\n{json.dumps(extraction)}\n\n"
                       f"Agent 2 risk findings:\n{json.dumps(risk_findings)}\n\n"
                       f"Agent 3 policy findings:\n{json.dumps(policy_findings)}"
                       if extraction
                       else f"Contract:\n{contract_text[:8000]}")
                )

                result = run_tool_loop(
                    client, SYSTEM_PROMPT, context, TOOLS, handle_tool
                )

                display_output(result)
                vendor = extraction.get("vendor_name") or "unknown"
                save_brief(result, vendor, contract_file)

                if is_collab:
                    sending("orchestrator", "Negotiation brief complete")
                    send_message(AGENT_NAME, "orchestrator",
                                 "negotiation_complete", {
                                     "contract_file":        contract_file,
                                     "negotiation_findings": result
                                 })
                else:
                    send_message(AGENT_NAME, "controller",
                                 "standalone_result", {
                                     "agent":       AGENT_NAME,
                                     "summary":     summary_for_controller(result),
                                     "full_result": result
                                 })
                log("\n  ✅ Done. Waiting...\n")

        time.sleep(1)


if __name__ == "__main__":
    if "--service" in sys.argv:
        run_as_service()
    else:
        contract_file = (sys.argv[1] if len(sys.argv) > 1
                         else prompt_contract_file())
        header("AGENT 4 — NEGOTIATION ADVISOR",
               f"Standalone: {os.path.basename(contract_file)}")
        log("Tools: read_contract | draft_redline | "
            "build_strategy | issue_negotiation_brief\n")
        contract_text           = util_read_contract(contract_file)
        _state["contract_text"] = contract_text
        result = run_tool_loop(
            client, SYSTEM_PROMPT,
            f"Negotiation brief for: {contract_file}\n\n"
            f"Contract:\n{contract_text[:8000]}",
            TOOLS, handle_tool
        )
        display_output(result)
        save_brief(result, "unknown", contract_file)
