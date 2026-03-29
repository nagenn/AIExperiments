"""
agent.py  —  Customer Support Triage Agent
==========================================
LangChain-powered agent that processes overnight support tickets,
looks up customer history, checks SLA compliance, and either drafts
a personalised response or escalates with a structured handoff note.

Prerequisites:
    1. Run seed_db.py once to populate support_history.db
    2. Run: uvicorn api:app --reload --port 8000  (in a separate terminal)
    3. Set OPENAI_API_KEY in your .env file

Run the demo:
    python agent.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TICKETS_CSV = os.path.join(BASE_DIR, "tickets.csv")
SLA_JSON    = os.path.join(BASE_DIR, "sla_guidelines.json")
API_BASE    = "http://localhost:8000"

SLA_GUIDELINES = json.loads(open(SLA_JSON, encoding="utf-8").read())

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def classify_ticket(ticket_id: str, subject: str, body: str) -> str:
    """
    Classify a support ticket into a category and tier.

    Returns:
      - category: the type of issue
      - tier: 1 (auto-resolvable), 2 (needs human review), 3 (urgent/escalate immediately)
      - reasoning: one-sentence explanation
    """
    categories = [
        "return_request", "billing_dispute", "technical_issue",
        "account_access", "delivery_complaint", "feature_request",
        "cancellation_request", "data_privacy_request", "general_enquiry"
    ]

    text = f"Subject: {subject}\n\nBody: {body}".lower()

    # Rule-based classification
    if any(w in text for w in ["gdpr", "delete", "data privacy", "right to be forgotten"]):
        category, tier = "data_privacy_request", 3
    elif any(w in text for w in ["cancel", "cancelling", "terminate", "end contract"]):
        category, tier = "cancellation_request", 3
    elif any(w in text for w in ["return", "refund", "money back"]):
        category, tier = "return_request", 1
    elif any(w in text for w in ["billing", "invoice", "charge", "overcharged", "credit"]):
        category, tier = "billing_dispute", 2
    elif any(w in text for w in ["locked out", "cannot access", "account", "password", "login", "reset"]):
        category, tier = "account_access", 2
    elif any(w in text for w in ["slow", "error", "bug", "not working", "broken", "api", "rate limit", "429"]):
        category, tier = "technical_issue", 2
    elif any(w in text for w in ["feature", "request", "suggestion", "would love", "would be great"]):
        category, tier = "feature_request", 1
    elif any(w in text for w in ["delivery", "shipping", "not arrived", "delayed"]):
        category, tier = "delivery_complaint", 2
    else:
        category, tier = "general_enquiry", 1

    reasoning_map = {
        "data_privacy_request": "Regulatory obligation — must be treated as critical regardless of customer tier.",
        "cancellation_request": "Revenue at risk — requires immediate account management involvement.",
        "return_request":       "Standard return request — check customer history before deciding.",
        "billing_dispute":      "Financial discrepancy — requires verification and potential credit note.",
        "account_access":       "Access issue — may be blocking customer's work, time-sensitive.",
        "technical_issue":      "Product issue — assess severity and customer tier to prioritise.",
        "feature_request":      "Enhancement request — log and acknowledge, no urgency.",
        "delivery_complaint":   "Fulfilment issue — check order status and SLA.",
        "general_enquiry":      "General question — standard response appropriate.",
    }

    return json.dumps({
        "ticket_id": ticket_id,
        "category": category,
        "tier": tier,
        "reasoning": reasoning_map.get(category, "Standard classification applied.")
    })


@tool
def lookup_customer(customer_id: str) -> str:
    """
    Look up a customer's full profile and interaction history summary
    from the Customer History API.

    Returns customer tier, contract value, SLA tier, total purchases,
    returns, disputes, churn signals, and last interaction.
    """
    try:
        summary_resp = requests.get(f"{API_BASE}/customers/{customer_id}/summary", timeout=5)
        if summary_resp.status_code == 404:
            return json.dumps({"error": f"Customer {customer_id} not found in the database."})
        summary_resp.raise_for_status()
        data = summary_resp.json()

        customer = data["customer"]
        stats    = data["stats"]
        last     = data["last_support_interaction"]

        # Calculate return rate
        purchases = stats["total_purchases"] or 0
        returns   = stats["total_returns"] or 0
        return_rate = round((returns / purchases * 100), 1) if purchases > 0 else 0

        # Churn risk flag
        churn_signals = []
        if stats["cancellation_requests"] > 0:
            churn_signals.append("has submitted a cancellation request before")
        if stats["negative_interactions"] >= 3:
            churn_signals.append(f"{stats['negative_interactions']} negative interactions on record")
        if stats["times_escalated"] >= 2:
            churn_signals.append(f"escalated {stats['times_escalated']} times previously")

        return json.dumps({
            "customer_id":        customer["customer_id"],
            "name":               customer["name"],
            "company":            customer["company"],
            "tier":               customer["tier"],
            "sla_tier":           customer["sla_tier"],
            "contract_value_usd": customer["contract_value"],
            "customer_since":     customer["join_date"],
            "country":            customer["country"],
            "total_purchases":    purchases,
            "total_returns":      returns,
            "return_rate_pct":    return_rate,
            "billing_disputes":   stats["billing_disputes"],
            "times_escalated":    stats["times_escalated"],
            "total_spend_usd":    stats["total_spend_usd"],
            "last_interaction":   last,
            "churn_risk_signals": churn_signals,
        })

    except requests.ConnectionError:
        return json.dumps({
            "error": "Cannot reach Customer History API. Is 'uvicorn api:app --reload' running?"
        })


@tool
def check_sla(customer_id: str, sla_tier: str, ticket_category: str, submitted_at: str) -> str:
    """
    Check whether a ticket is within SLA, at risk, or already breached.

    Args:
        customer_id:      the customer ID
        sla_tier:         Platinum / Gold / Silver / Bronze
        ticket_category:  the ticket type from classify_ticket
        submitted_at:     ISO datetime string when the ticket was submitted

    Returns SLA status, hours elapsed, threshold, and urgency level.
    """
    now         = datetime.now()
    submitted   = datetime.fromisoformat(submitted_at)
    hours_open  = round((now - submitted).total_seconds() / 3600, 1)

    # Get base SLA for tier
    tier_sla = SLA_GUIDELINES["sla_tiers"].get(sla_tier, SLA_GUIDELINES["sla_tiers"]["Bronze"])
    response_threshold = tier_sla["first_response_hours"]
    resolution_threshold = tier_sla["resolution_hours"]

    # Check for category override
    override = SLA_GUIDELINES["ticket_type_modifiers"].get(ticket_category, {})
    if override.get("override_priority") == "critical":
        response_threshold  = min(response_threshold, 2)
        resolution_threshold = override.get("max_resolution_hours", resolution_threshold)
        override_note = override.get("note", "")
    else:
        override_note = override.get("note", "")

    # Determine status
    if hours_open > resolution_threshold:
        status   = "BREACHED"
        urgency  = "CRITICAL"
    elif hours_open > response_threshold:
        status   = "RESPONSE_DUE"
        urgency  = "HIGH"
    elif hours_open > response_threshold * 0.75:
        status   = "AT_RISK"
        urgency  = "MEDIUM"
    else:
        status   = "WITHIN_SLA"
        urgency  = "NORMAL"

    return json.dumps({
        "customer_id":            customer_id,
        "sla_tier":               sla_tier,
        "hours_open":             hours_open,
        "first_response_threshold_hours": response_threshold,
        "resolution_threshold_hours":     resolution_threshold,
        "sla_status":             status,
        "urgency":                urgency,
        "category_override_note": override_note,
    })


@tool
def draft_response(
    ticket_id: str,
    customer_name: str,
    customer_tier: str,
    ticket_category: str,
    ticket_subject: str,
    ticket_body: str,
    customer_history_summary: str,
    sla_status: str,
) -> str:
    """
    Draft a personalised, tier-aware support email response.

    This tool uses the customer's full history and context to write a
    professional email that reflects the customer's relationship with us —
    not a generic template. It adapts tone, authority level, and content
    based on tier, history, and whether this is a repeat issue.

    Use for Tier-1 tickets and Tier-2 tickets where a response is appropriate.
    Do NOT use for Tier-3 escalations — use draft_escalation_note instead.
    """
    from openai import OpenAI
    oai = OpenAI()

    tone_map = {
        "Enterprise": "warm, senior, highly personal — treat them as a VIP",
        "Business":   "professional and attentive — they are valued customers",
        "Standard":   "friendly and helpful — clear and efficient",
        "Trial":      "encouraging and helpful — we want to convert them",
    }
    tone = tone_map.get(customer_tier, "professional and helpful")

    prompt = f"""
You are a senior customer support specialist. Write a professional email response to this support ticket.

TICKET DETAILS:
  Ticket ID:  {ticket_id}
  Category:   {ticket_category}
  Subject:    {ticket_subject}
  Body:       {ticket_body}

CUSTOMER PROFILE:
  Name:    {customer_name}
  Tier:    {customer_tier}
  History: {customer_history_summary}

SLA STATUS: {sla_status}

TONE GUIDANCE: {tone}

INSTRUCTIONS:
- Address the customer by first name
- Acknowledge their specific issue directly — do not be generic
- Reference their history where relevant (e.g. long-standing customer, first issue, repeat issue)
- If SLA is BREACHED or RESPONSE_DUE, acknowledge the delay and apologise
- Provide a clear next step or resolution
- If this is a return request: check history — approve warmly if first return with good purchase history,
  or acknowledge carefully if they have a high return rate (do not approve outright)
- Keep to 150 words maximum
- End with a professional sign-off from "The Support Team"

Write ONLY the email body (no subject line, no metadata). Start with "Dear [First Name],"
"""

    response = oai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


@tool
def draft_escalation_note(
    ticket_id: str,
    customer_name: str,
    customer_tier: str,
    ticket_category: str,
    ticket_subject: str,
    escalation_reason: str,
    sla_status: str,
    contract_value_usd: int,
) -> str:
    """
    Draft a structured internal escalation note for Tier-2/3 tickets.

    Use this for tickets that require human review, urgent action,
    or account management involvement.
    """
    urgency_map = {
        "cancellation_request": "🚨 CRITICAL — Revenue at risk",
        "data_privacy_request": "🚨 CRITICAL — Regulatory obligation",
        "account_access":       "🔴 HIGH — Customer blocked",
        "billing_dispute":      "🔴 HIGH — Financial dispute",
        "technical_issue":      "🟡 MEDIUM — Product issue",
        "return_request":       "🟡 MEDIUM — Return review required",
        "delivery_complaint":   "🟡 MEDIUM — Fulfilment issue",
        "general_enquiry":      "🟢 LOW — Standard review",
    }
    urgency = urgency_map.get(ticket_category, "🟡 MEDIUM — Review required")

    assign_map = {
        "cancellation_request": "Senior Account Manager",
        "data_privacy_request": "Legal & Compliance Team",
        "billing_dispute":      "Finance & Billing Team",
        "account_access":       "Platform Engineering — Tier 2",
        "technical_issue":      "Technical Support — Tier 2",
        "return_request":       "Customer Success Manager",
        "delivery_complaint":   "Fulfilment Operations",
    }
    assign_to = assign_map.get(ticket_category, "Tier-2 Support Team")

    note = f"""
┌─────────────────────────────────────────────────────────────┐
│  ESCALATION NOTE — {ticket_id:<42}│
└─────────────────────────────────────────────────────────────┘
  Urgency      : {urgency}
  Assign To    : {assign_to}
  Customer     : {customer_name} ({customer_tier} tier)
  Contract Val : ${contract_value_usd:,} / year
  Category     : {ticket_category}
  SLA Status   : {sla_status}

  Subject      : {ticket_subject}

  Reason for escalation:
  {escalation_reason}

  ACTION REQUIRED:
  - Review ticket and make contact within SLA window
  - Log outcome in CRM
  - If cancellation risk: loop in Account Director
└─────────────────────────────────────────────────────────────┘"""
    return note


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

def build_agent() -> AgentExecutor:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = hub.pull("hwchase17/openai-tools-agent")
    tools = [classify_ticket, lookup_customer, check_sla, draft_response, draft_escalation_note]
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

DIVIDER     = "=" * 66
SUB_DIVIDER = "─" * 66

STATUS_ICONS = {
    "WITHIN_SLA":    "✅",
    "AT_RISK":       "⚠️ ",
    "RESPONSE_DUE":  "🔴",
    "BREACHED":      "🚨",
}

ACTION_ICONS = {
    1: "✅  RESOLVE",
    2: "🔺  ESCALATE",
    3: "🚨  URGENT ESCALATE",
}


def print_header(total: int):
    print(f"\n{DIVIDER}")
    print("  CUSTOMER SUPPORT TRIAGE AGENT")
    print(f"  Processing {total} tickets from overnight queue")
    print(f"  {datetime.now().strftime('%A, %d %B %Y — %H:%M')}")
    print(DIVIDER)


def print_ticket_result(idx: int, total: int, ticket: dict, result: str):
    print(f"\n{DIVIDER}")
    print(f"  [{idx} of {total}]  Ticket {ticket['ticket_id']}  —  {ticket['subject']}")
    print(SUB_DIVIDER)
    print(result)
    print(SUB_DIVIDER)


def print_summary(results: list[dict]):
    resolved   = [r for r in results if r["tier"] == 1]
    escalated  = [r for r in results if r["tier"] == 2]
    urgent     = [r for r in results if r["tier"] == 3]
    sla_issues = [r for r in results if r.get("sla_status") in ("BREACHED", "RESPONSE_DUE")]

    print(f"\n{DIVIDER}")
    print("  TRIAGE COMPLETE — SUMMARY")
    print(DIVIDER)
    print(f"  Total tickets processed  : {len(results)}")
    print(f"  ✅  Resolved  (Tier 1)   : {len(resolved)}")
    print(f"  🔺  Escalated (Tier 2)   : {len(escalated)}")
    print(f"  🚨  Urgent    (Tier 3)   : {len(urgent)}")
    print(f"  🕐  SLA issues flagged   : {len(sla_issues)}")

    if sla_issues:
        print(f"\n  SLA ALERTS:")
        for r in sla_issues:
            icon = STATUS_ICONS.get(r.get("sla_status", ""), "⚠️ ")
            print(f"    {icon}  {r['ticket_id']}  —  {r['customer_name']}  ({r.get('sla_status','?')})")

    print(DIVIDER)


# ---------------------------------------------------------------------------
# Main demo loop
# ---------------------------------------------------------------------------

AGENT_PROMPT_TEMPLATE = """
You are a customer support triage agent. Process this support ticket step by step.

TICKET:
  ID          : {ticket_id}
  Customer ID : {customer_id}
  Submitted   : {submitted_at}
  Subject     : {subject}
  Body        : {body}

Follow these steps IN ORDER — call each tool in sequence:

STEP 1 — Call classify_ticket with the ticket_id, subject and body.

STEP 2 — Call lookup_customer with the customer_id to get their full history.

STEP 3 — Call check_sla using the customer's sla_tier from Step 2,
         the ticket category from Step 1, and the submitted_at time.

STEP 4 — Based on Steps 1-3, decide the action:
  - Tier 1 (auto-resolve): call draft_response with full context
  - Tier 2 or 3 (escalate): call draft_escalation_note with full context

    Special rules (override tier if needed):
    * return_request + prior returns >= 2  → escalate (Tier 2)
    * return_request + prior returns == 0  → resolve  (Tier 1)
    * cancellation_request                 → always Tier 3
    * data_privacy_request                 → always Tier 3
    * SLA BREACHED + Enterprise/Platinum   → upgrade to Tier 3

STEP 5 — Write a concise triage summary in this exact format:

TRIAGE RESULT
  Ticket      : {ticket_id}
  Customer    : [name] | [tier] | [SLA tier]
  Category    : [category]
  Tier        : [1/2/3]
  SLA Status  : [status] ([hours] hrs open)
  Key Finding : [one sentence about what the customer history revealed that influenced the decision]
  Action      : [RESOLVE / ESCALATE / URGENT ESCALATE]

[paste the full drafted response or escalation note here]
"""


def run_demo():
    print_header(0)  # Will reprint with count

    # Load tickets
    if not os.path.exists(TICKETS_CSV):
        print(f"ERROR: tickets.csv not found at {TICKETS_CSV}")
        return

    tickets = pd.read_csv(TICKETS_CSV).to_dict(orient="records")
    total   = len(tickets)

    print_header(total)

    executor = build_agent()
    results  = []

    for idx, ticket in enumerate(tickets, start=1):
        prompt = AGENT_PROMPT_TEMPLATE.format(**ticket)

        print(f"\n⏳  Processing ticket {idx}/{total}: {ticket['ticket_id']}...")
        print("    (Agent reasoning trace follows)\n")

        try:
            response = executor.invoke({"input": prompt})
            output   = response.get("output", str(response))
        except Exception as e:
            output = f"ERROR processing ticket: {e}"

        print_ticket_result(idx, total, ticket, output)

        # Store result metadata for summary (best-effort parse)
        result_meta = {
            "ticket_id":     ticket["ticket_id"],
            "customer_name": ticket.get("customer_id", "Unknown"),
            "tier":          2,  # default
            "sla_status":    "UNKNOWN",
        }
        if "Tier        : 1" in output:
            result_meta["tier"] = 1
        elif "Tier        : 3" in output:
            result_meta["tier"] = 3

        for status in ("BREACHED", "RESPONSE_DUE", "AT_RISK", "WITHIN_SLA"):
            if status in output:
                result_meta["sla_status"] = status
                break

        results.append(result_meta)

        if idx < total:
            input("\n  ▶  Press Enter to continue to the next ticket...")

    print_summary(results)
    print("\n  Demo complete. All tickets processed.\n")


if __name__ == "__main__":
    run_demo()
