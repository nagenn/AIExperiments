"""
orchestrator.py
---------------
Manages the full collaborative review.
Called by demo.py option 5. Not run directly.
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import header, section, log, sending, get_client, unwrap_result
from message_bus import send_message, wait_for_message, clear_all

AGENT_NAME = "orchestrator"
client     = get_client()


def assess_urgency(risk_findings):
    overall  = risk_findings.get("overall_risk", "GREEN")
    critical = sum(
        1 for k in ["liability_risk", "data_risk", "ip_risk", "payment_risk"]
        if risk_findings.get(k) in ["CRITICAL", "HIGH"]
    )
    if overall == "RED" and critical >= 3:
        return "ESCALATE"
    return "FULL_REVIEW"


def run(contract_file, contract_text):
    header("ORCHESTRATOR — COLLABORATIVE REVIEW",
           f"Contract: {os.path.basename(contract_file)}")
    clear_all()
    log("Message bus cleared.\n")

    # ── Step 1: Contract Intelligence ─────────────────────────────────────────
    section("Step 1 of 4 — Contract Intelligence")
    sending("contract_intelligence", "Sending contract")
    send_message(AGENT_NAME, "contract_intelligence", "collab_request", {
        "contract_file": contract_file,
        "contract_text": contract_text
    })
    log("  Waiting for Agent 1...")
    msg1       = wait_for_message(AGENT_NAME, "extraction_complete", timeout=300)
    extraction = unwrap_result(msg1["content"]["extraction"])
    log("  ✅ Agent 1 complete.")

    # ── Step 2: Risk & Benchmarking ───────────────────────────────────────────
    section("Step 2 of 4 — Risk & Benchmarking")
    sending("risk_benchmarking", "Sending extraction from Agent 1")
    send_message(AGENT_NAME, "risk_benchmarking", "collab_request", {
        "contract_file": contract_file,
        "contract_text": contract_text,
        "extraction":    extraction
    })
    log("  Waiting for Agent 2...")
    msg2          = wait_for_message(AGENT_NAME, "risk_complete", timeout=300)
    risk_findings = unwrap_result(msg2["content"]["risk_findings"])
    log("  ✅ Agent 2 complete.")

    urgency = assess_urgency(risk_findings)
    overall = risk_findings.get("overall_risk", "?")
    log(f"\n  🎯 Risk: {overall} — Decision: {urgency}")
    if urgency == "ESCALATE":
        log("  🎯 CRITICAL threshold. Escalating.")

    # ── Step 3: Policy & Compliance ───────────────────────────────────────────
    section("Step 3 of 4 — Policy & Compliance")
    sending("policy_compliance", "Sending risk findings — Agent 3 will challenge inconsistencies")
    send_message(AGENT_NAME, "policy_compliance", "collab_request", {
        "contract_file": contract_file,
        "contract_text": contract_text,
        "extraction":    extraction,
        "risk_findings": risk_findings
    })
    log("  Waiting for Agent 3 (may challenge Agent 2)...")
    msg3            = wait_for_message(AGENT_NAME, "policy_complete", timeout=300)
    policy_findings = unwrap_result(msg3["content"]["policy_findings"])
    log("  ✅ Agent 3 complete.")

    # ── Step 4: Negotiation Advisor ───────────────────────────────────────────
    section("Step 4 of 4 — Negotiation Advisor")
    sending("negotiation_advisor", "Sending full picture — all agent findings")
    send_message(AGENT_NAME, "negotiation_advisor", "collab_request", {
        "contract_file":   contract_file,
        "contract_text":   contract_text,
        "extraction":      extraction,
        "risk_findings":   risk_findings,
        "policy_findings": policy_findings
    })
    log("  Waiting for Agent 4...")
    msg4                 = wait_for_message(AGENT_NAME, "negotiation_complete",
                                             timeout=300)
    negotiation_findings = unwrap_result(msg4["content"]["negotiation_findings"])
    log("  ✅ Agent 4 complete.")

    # ── Summary ───────────────────────────────────────────────────────────────
    vendor = extraction.get("vendor_name", "unknown")
    section("REVIEW COMPLETE")
    log(f"  • Vendor  : {vendor}")
    log(f"  • Risk    : {risk_findings.get('overall_risk','?')}")
    log(f"  • Policy  : {policy_findings.get('overall_verdict','?')}")
    log(f"  • Action  : "
        f"{negotiation_findings.get('negotiation_verdict','?').replace('_',' ')}")
    log(f"  • Redlines: "
        f"{len(negotiation_findings.get('redline_suggestions',[]))} clauses\n")

    return {
        "extraction":      extraction,
        "risk_findings":   risk_findings,
        "policy_findings": policy_findings,
        "negotiation":     negotiation_findings,
        "vendor":          vendor
    }
