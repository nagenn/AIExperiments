"""
agents/policy_compliance.py
----------------------------
AGENT 3 — POLICY & COMPLIANCE AGENT

Tools:
  • read_contract        — read the contract independently
  • check_policy_rule    — check a clause against a policy rule
  • challenge_risk_agent — send a challenge to Agent 2 (collab mode only)
  • issue_verdict        — issue the final policy verdict
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (get_client, read_contract as util_read_contract,
                   prompt_contract_file, header, section, log,
                   sending, receiving,
                   challenge_log, resolved_log,
                   unwrap_result, run_tool_loop, SILENT)
from message_bus import send_message, get_messages, mark_read, wait_for_message

AGENT_NAME = "policy_compliance"
client     = get_client()

POLICY_RULES = {
    "RULE_1":  {"name": "Liability Cap",      "rule": "Minimum 6 months fees. Below 6 months = AUTOMATIC RED."},
    "RULE_2":  {"name": "Payment Terms",      "rule": "Maximum Net 30. 60+ days = AUTOMATIC RED."},
    "RULE_3":  {"name": "Data Usage Rights",  "rule": "No vendor data usage beyond service delivery = AUTOMATIC RED."},
    "RULE_4":  {"name": "IP Ownership",       "rule": "All outputs using customer data belong to customer = AUTOMATIC RED."},
    "RULE_5":  {"name": "Termination Notice", "rule": "Customer notice max 90 days. No post-termination liability."},
    "RULE_6":  {"name": "Data Return",        "rule": "Full export within 30 days on exit = AUTOMATIC RED."},
    "RULE_7":  {"name": "Governing Law",      "rule": "Acceptable: India, UK, US-NY, Singapore. Foreign language = not acceptable."},
    "RULE_8":  {"name": "Amendment Rights",   "rule": "Minimum 30 days notice = AUTOMATIC RED below 30 days."},
    "RULE_9":  {"name": "SLA",                "rule": "Minimum 99.5%. Below 99.0% = AUTOMATIC RED."},
    "RULE_10": {"name": "Assignment",         "rule": "Vendor cannot assign without customer consent."},
}

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
            "name": "check_policy_rule",
            "description": "Check a specific clause against an organisational policy rule.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_id": {
                        "type": "string",
                        "enum": list(POLICY_RULES.keys())
                    },
                    "clause_value":     {"type": "string"},
                    "clause_reference": {"type": "string"}
                },
                "required": ["rule_id", "clause_value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "challenge_risk_agent",
            "description": (
                "Challenge Agent 2 when their rating conflicts with policy. "
                "Use when Risk Agent rated AMBER but policy says AUTOMATIC RED."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_id":                {"type": "string"},
                    "finding":                {"type": "string"},
                    "risk_agent_rating":      {"type": "string"},
                    "policy_required_rating": {"type": "string"},
                    "challenge_message":      {"type": "string"}
                },
                "required": ["rule_id", "finding", "challenge_message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "issue_verdict",
            "description": "Issue the final policy verdict after all checks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": ["APPROVED", "CONDITIONAL", "REJECTED"]
                    },
                    "reason":     {"type": "string"},
                    "conditions": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["verdict", "reason"]
            }
        }
    }
]

SYSTEM_PROMPT = f"""
You are the Policy & Compliance Agent.

Use tools in this order:
1. read_contract (skip if risk findings provided)
2. check_policy_rule — check the MOST RELEVANT rules only (3-4 calls max)
3. challenge_risk_agent — MANDATORY if any conflict found (see below)
4. issue_verdict — after all checks

Focus check_policy_rule on the highest-risk areas: liability cap, payment terms,
data usage rights, SLA. Skip rules where the contract clearly complies.

POLICY RULES:
{json.dumps(POLICY_RULES, indent=2)}

CHALLENGE RULE — THIS IS MANDATORY:
After checking policy rules, compare your findings against the Risk Agent's
ratings in the risk_findings provided. You MUST call challenge_risk_agent if:
  - Policy says AUTOMATIC RED but Risk Agent rated the area AMBER or MEDIUM
  - Risk Agent missed a breach that policy rules clearly identify
  - Risk Agent's overall_risk is AMBER but you find AUTOMATIC RED breaches

Do not skip the challenge. The audience expects to see Agent 3 challenge
Agent 2 when their ratings conflict with policy. This is a key part of the demo.

IMPORTANT: Return your final answer as a raw JSON object only.
No preamble, no explanation, no markdown fences. Just the JSON.

Final answer format:
{{
  "overall_verdict": "APPROVED|CONDITIONAL|REJECTED",
  "policy_breaches": [
    {{
      "rule": "",
      "clause_reference": "",
      "finding": "",
      "severity": "AUTOMATIC_RED|HIGH|MEDIUM",
      "challenge_sent": false
    }}
  ],
  "policy_passes": [],
  "conditions_for_approval": []
}}
"""

SYSTEM_PROMPT_STANDALONE = f"""
You are the Policy & Compliance Agent.

Use tools in this order:
1. read_contract
2. check_policy_rule — check the MOST RELEVANT rules only (3-4 calls max)
3. issue_verdict — after all checks

Focus check_policy_rule on the highest-risk areas: liability cap, payment terms,
data usage rights, SLA. Skip rules where the contract clearly complies.

Do NOT call challenge_risk_agent in standalone mode.

POLICY RULES:
{json.dumps(POLICY_RULES, indent=2)}

IMPORTANT: Return your final answer as a raw JSON object only.
No preamble, no explanation, no markdown fences. Just the JSON.

Final answer format:
{{
  "overall_verdict": "APPROVED|CONDITIONAL|REJECTED",
  "policy_breaches": [
    {{
      "rule": "",
      "clause_reference": "",
      "finding": "",
      "severity": "AUTOMATIC_RED|HIGH|MEDIUM",
      "challenge_sent": false
    }}
  ],
  "policy_passes": [],
  "conditions_for_approval": []
}}
"""

_state = {"contract_text": "", "challenges_sent": [], "is_collab": False}


def handle_tool(name, args):
    if name == "read_contract":
        try:
            text = util_read_contract(args.get("filepath", ""))
            _state["contract_text"] = text
            return "Contract read."
        except Exception as e:
            return f"Error: {e}"

    elif name == "check_policy_rule":
        rule_id    = args.get("rule_id", "")
        clause_val = args.get("clause_value", "")
        clause_ref = args.get("clause_reference", "")
        rule       = POLICY_RULES.get(rule_id, {})
        if not rule:
            return f"Unknown rule: {rule_id}"
        return (f"{rule_id} ({rule['name']}): "
                f"Contract='{clause_val}'. Rule: {rule['rule']} "
                f"Ref: {clause_ref or 'unspecified'}")

    elif name == "challenge_risk_agent":
        if not _state.get("is_collab"):
            return SILENT
        challenge_data = {
            "rule":                   args.get("rule_id", ""),
            "finding":                args.get("finding", ""),
            "risk_agent_rating":      args.get("risk_agent_rating", ""),
            "policy_required_rating": args.get("policy_required_rating", ""),
            "challenge":              args.get("challenge_message", "")
        }
        _state["challenges_sent"].append(challenge_data)
        challenge_log(f"{args.get('rule_id')}: "
                      f"{args.get('challenge_message','')[:90]}")
        return f"Challenge queued for {args.get('rule_id')}."

    elif name == "issue_verdict":
        verdict = args.get("verdict", "")
        icons   = {"APPROVED": "✅", "CONDITIONAL": "🟡", "REJECTED": "🔴"}
        log(f"\n  {icons.get(verdict,'⚪')}  VERDICT: {verdict}")
        log(f"  {args.get('reason','')[:120]}")
        return f"Verdict: {verdict}."

    return f"Unknown tool: {name}"


def display_output(result):
    result  = unwrap_result(result)

    # Safety net — if parsing still failed, show raw output rather than "?"
    if "raw_output" in result:
        section("POLICY & COMPLIANCE — VERDICT")
        log("⚠️  Could not parse structured output. Raw response:")
        log(result["raw_output"][:500])
        return

    section("POLICY & COMPLIANCE — VERDICT")

    verdict = result.get("overall_verdict", "?")
    icons   = {"APPROVED": "✅", "CONDITIONAL": "🟡", "REJECTED": "🔴"}
    icon    = icons.get(verdict, "⚪")
    print(f"\n  {icon}  VERDICT: {verdict}\n")

    breaches = result.get("policy_breaches", [])
    if breaches:
        print(f"  Policy breaches ({len(breaches)}):")
        for b in breaches:
            sev  = b.get("severity", "")
            icon = "🔴" if "RED" in sev else "🟡"
            print(f"  {icon} {b.get('rule','')}: {b.get('finding','')[:80]}")

    if result.get("policy_passes"):
        print(f"\n  Policy passes:")
        for p in result["policy_passes"]:
            print(f"  ✅ {p}")

    if result.get("conditions_for_approval"):
        section("CONDITIONS FOR APPROVAL")
        for c in result["conditions_for_approval"]:
            log(f"  • {c}")


def summary_for_controller(result):
    result  = unwrap_result(result)
    verdict = result.get("overall_verdict", "?")
    icons   = {"APPROVED": "✅", "CONDITIONAL": "🟡", "REJECTED": "🔴"}
    icon    = icons.get(verdict, "⚪")
    lines   = [f"  {icon}  Policy verdict: {verdict}"]
    breaches  = result.get("policy_breaches", [])
    auto_reds = [b for b in breaches if "RED" in b.get("severity", "")]
    if auto_reds:
        lines.append(f"  🔴  Automatic RED triggers: {len(auto_reds)}")
    for b in breaches[:3]:
        lines.append(f"  •  {b.get('rule','')}: {b.get('finding','')[:80]}")
    return "\n".join(lines)


def run_as_service():
    header("AGENT 3 — POLICY & COMPLIANCE",
           "Tools: read_contract | check_policy_rule | "
           "challenge_risk_agent | issue_verdict")
    log("Waiting for contracts...\n")

    while True:
        for msg_type in ("standalone_request", "collab_request"):
            msgs = get_messages(AGENT_NAME, message_type=msg_type)
            for filepath, msg in msgs:
                mark_read(filepath)
                contract_file  = msg["content"]["contract_file"]
                contract_text  = msg["content"]["contract_text"]
                extraction     = unwrap_result(msg["content"].get("extraction", {}))
                risk_findings  = unwrap_result(msg["content"].get("risk_findings", {}))
                is_collab      = (msg_type == "collab_request")

                _state["contract_text"]   = contract_text
                _state["challenges_sent"] = []
                _state["is_collab"]       = is_collab

                header("AGENT 3 — POLICY & COMPLIANCE",
                       f"{'Collab' if is_collab else 'Standalone'}: "
                       f"{os.path.basename(contract_file)}")

                context = (
                    f"Policy check: {contract_file}\n\n"
                    + (f"Risk findings:\n{json.dumps(risk_findings)}\n\n"
                       f"Extraction:\n{json.dumps(extraction)}"
                       if risk_findings
                       else f"Contract:\n{contract_text[:8000]}")
                )

                result = run_tool_loop(
                    client,
                    SYSTEM_PROMPT if is_collab else SYSTEM_PROMPT_STANDALONE,
                    context, TOOLS, handle_tool
                )

                # ── Send challenges and handle responses (collab only) ─────────
                challenge_resolutions = []
                if _state["challenges_sent"] and is_collab:
                    for ch in _state["challenges_sent"]:
                        send_message(AGENT_NAME, "risk_benchmarking",
                                     "policy_challenge", ch)
                    log(f"\n  Waiting for Agent 2 — "
                        f"{len(_state['challenges_sent'])} challenge(s)...")

                    for _ in _state["challenges_sent"]:
                        try:
                            resp         = wait_for_message(AGENT_NAME,
                                                            "challenge_response",
                                                            timeout=120)
                            resp_content = unwrap_result(resp["content"])
                            decision     = resp_content.get("decision", "defend")
                            rule         = resp_content.get("rule", "")
                            agent2_response = (resp_content.get("response", {})
                                                           .get("response", "")
                                               if isinstance(resp_content.get("response"), dict)
                                               else str(resp_content.get("response", "")))
                            receiving("risk_benchmarking", agent2_response[:80])

                            if decision == "upgrade":
                                resolved_log(f"Agent 2 agreed to upgrade on {rule}.")
                                for breach in result.get("policy_breaches", []):
                                    if breach.get("rule") == rule:
                                        breach["challenge_sent"] = True
                                        breach["finding"] += (
                                            " [UPGRADED after Agent 2 conceded]")
                                result["overall_verdict"] = "REJECTED"
                                challenge_resolutions.append(("upgrade", rule, agent2_response))
                            else:
                                log(f"  Agent 2 defended on {rule}. Policy ruling stands.")
                                challenge_resolutions.append(("defend", rule, agent2_response))

                        except TimeoutError:
                            log("  No response. Policy ruling stands.")
                            challenge_resolutions.append(("timeout", "", ""))

                # ── Show all challenge outcomes clearly before final verdict ──
                if challenge_resolutions:
                    print(f"\n  {'─'*50}")
                    for decision, rule, response in challenge_resolutions:
                        if decision == "upgrade":
                            print(f"  ⚡ CHALLENGE RESOLVED — Agent 2 CONCEDED on {rule}")
                            print(f"     Verdict upgraded to REJECTED.")
                            if response:
                                print(f"     Agent 2 said: {response[:100]}")
                        elif decision == "defend":
                            print(f"  ⚡ CHALLENGE RESOLVED — Agent 2 DEFENDED on {rule}")
                            print(f"     Policy ruling stands.")
                            if response:
                                print(f"     Agent 2 said: {response[:100]}")
                        else:
                            print(f"  ⚡ CHALLENGE TIMED OUT — no response from Agent 2.")
                            print(f"     Policy ruling stands.")
                    print(f"  {'─'*50}")

                display_output(result)

                if is_collab:
                    sending("orchestrator", "Policy check complete")
                    send_message(AGENT_NAME, "orchestrator", "policy_complete", {
                        "contract_file":   contract_file,
                        "contract_text":   contract_text,
                        "extraction":      extraction,
                        "risk_findings":   risk_findings,
                        "policy_findings": result,
                    })
                else:
                    send_message(AGENT_NAME, "controller", "standalone_result", {
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
        header("AGENT 3 — POLICY & COMPLIANCE",
               f"Standalone: {os.path.basename(contract_file)}")
        log("Tools: read_contract | check_policy_rule | issue_verdict\n")
        _state["is_collab"]     = False
        contract_text           = util_read_contract(contract_file)
        _state["contract_text"] = contract_text
        result = run_tool_loop(
            client, SYSTEM_PROMPT_STANDALONE,
            f"Policy check: {contract_file}\n\nContract:\n{contract_text[:8000]}",
            TOOLS, handle_tool
        )
        display_output(result)
