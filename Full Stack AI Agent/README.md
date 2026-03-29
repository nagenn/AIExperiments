# Customer Support Triage Agent
### A LangChain Demo for Senior Executives

---

## What This Demo Shows

An AI agent that processes an overnight backlog of support tickets autonomously.
For each ticket, it:

1. **Classifies** the issue type and urgency tier
2. **Looks up** the customer's full history from a real REST API backed by SQLite
3. **Checks SLA** compliance against a policy JSON file
4. **Decides** — resolve, escalate, or urgent escalate — based on history + policy
5. **Drafts** either a personalised customer email or a structured internal escalation note

The "wow" moment: the agent catches nuance a human might miss —
a ticket that looks routine but belongs to a high-value Enterprise customer
already at SLA breach, or a return request from a customer who has
returned 2 out of 5 purchases (escalate) vs. a loyal customer making
their first-ever return (approve warmly).

---

## Demo Narrative (say this before you hit Enter)

> *"We have a backlog of support tickets that came in overnight.
> Right now, a human triage agent works through these one by one —
> it takes the better part of a morning.
> We have customers on 1-hour SLA commitments.
> By the time a human even reads the first ticket, we may have already
> burned through our response window.
> Watch what happens when the agent does this instead."*

Then run: `./launch_demo.sh`

---

## File Structure

```
support_triage_agent/
├── launch_demo.sh      ← START HERE — launches both terminal windows automatically
├── agent.py            ← Main LangChain agent (run this for the demo)
├── api.py              ← FastAPI REST wrapper over SQLite
├── seed_db.py          ← One-time setup: generates data with GPT + loads DB
├── tickets.csv         ← The 3 incoming tickets to process in the demo
├── sla_guidelines.json ← SLA thresholds and return policy rules
├── support_history.db  ← SQLite DB (created by seed_db.py)
└── README.md           ← This file
```

---

## Setup (one time only)

### 1. Install dependencies

```bash
pip install langchain langchain-openai openai fastapi uvicorn pandas requests python-dotenv
```

> **Note on SQLite:** No installation needed — `sqlite3` is built into Python's standard library. It will work out of the box.

### 2. Set up your .env file

Create a `.env` file in the same directory:

```
OPENAI_API_KEY=sk-...
```

### 3. Seed the database

This generates 50 customers and 500+ interaction records using GPT,
then loads them into SQLite. Takes about 60-90 seconds.

```bash
python3.9 seed_db.py
```

You should see:
```
✅ 50 customers generated
✅ 487 interactions generated
✅ Database ready.
```

---

## Running the Demo

### The easy way — use the launch script

```bash
chmod +x launch_demo.sh   # first time only
./launch_demo.sh
```

This will:
1. Check your `OPENAI_API_KEY` is set
2. Check the database exists (and offer to seed it if not)
3. Open **Terminal 1** automatically — starts the Customer History API
4. Open **Terminal 2** automatically — runs the triage agent

Press **Enter** in Terminal 2 to advance to the next ticket.

### Manual launch (if needed)

**Terminal 1 — Start the Customer History API:**
```bash
uvicorn api:app --reload --port 8000
```

**Terminal 2 — Run the Agent:**
```bash
python3.9 agent.py
```

---

## What to Watch For During the Demo

| Ticket | What the agent reveals |
|--------|----------------------|
| TKT-1003 | GDPR deletion request — auto-escalated regardless of tier, regulatory override in action |
| TKT-1006 | Third return request — agent spots the pattern in history and escalates instead of approving |
| TKT-1008 | Cancellation threat from Enterprise customer — highest urgency, Senior Account Manager assigned |

---

## API Endpoints (for reference)

With the API running, you can also browse the data manually:

- http://localhost:8000/customers/CUST-001
- http://localhost:8000/customers/CUST-001/summary
- http://localhost:8000/customers/CUST-001/history
- http://localhost:8000/docs  ← Interactive API docs (Swagger UI)

---

## Tools Summary

| Tool | Description |
|------|-------------|
| `classify_ticket` | Classifies ticket into category and tier 1/2/3 |
| `lookup_customer` | Calls the REST API to get customer profile + history stats |
| `check_sla` | Calculates SLA status from ticket age + customer tier |
| `draft_response` | GPT-written personalised email for Tier-1 resolutions |
| `draft_escalation_note` | Structured internal handoff note for Tier-2/3 tickets |
