# Contract Review — Multi-Agent AI Demo

A multi-agent AI system that reviews vendor contracts using four autonomous specialist agents. Built on plain Python and the OpenAI API — no frameworks, no abstractions. Every tool call, agent decision, and inter-agent message is visible in the terminals as it happens.

---

## What It Does

Four AI agents collaborate to review a contract, each playing a distinct role:

| Terminal | Agent | Role | Tools |
|---|---|---|---|
| 1 | Contract Intelligence | Reads and structures the contract | read_contract, identify_clauses, flag_ambiguity, summarise_plain_english |
| 2 | Risk & Benchmarking | Rates risk against industry benchmarks | read_contract, benchmark_clause, request_clarification, flag_risk |
| 3 | Policy & Compliance | Checks against internal policy rules | read_contract, check_policy_rule, challenge_risk_agent, issue_verdict |
| 4 | Negotiation Advisor | Drafts redlines and negotiation strategy | read_contract, draft_redline, build_strategy, issue_negotiation_brief |
| 5 | Demo Controller | Menu-driven interface for the audience | — |

---

## Prerequisites

Python 3.9 or higher. Then:

```bash
pip install openai PyPDF2
export OPENAI_API_KEY=your-key-here
chmod +x start_demo.sh
```

---

## Project Structure

```
demo.py                          — Terminal 5 controller (menu-driven)
agents/
  contract_intelligence.py       — Agent 1
  risk_benchmarking.py           — Agent 2
  policy_compliance.py           — Agent 3
  negotiation_advisor.py         — Agent 4
orchestrator.py                  — Collaborative workflow manager
message_bus.py                   — File-based inter-agent messaging
utils.py                         — Shared utilities, JSON parsing, tool-use loop
start_demo.sh                    — One-command launcher (opens 5 terminals)
```

Sample contracts (place in project root):
```
sample_contract.pdf              — DataSync Solutions Pte. Ltd. (all red flags)
contract_B_datasync.pdf          — Same vendor, second contract
contract_C_nexahr.txt            — Different vendor, mostly clean (AMBER outcome)
```

---

## Running the Demo

```bash
export OPENAI_API_KEY=your-key-here
./start_demo.sh
```

This opens all 5 terminals automatically. Use **Terminal 5** to drive the demo.

To start fresh between runs:

```bash
rm -rf messages/ negotiation_briefs/
```

---

## Demo Menu (Terminal 5)

```
  PART 1 — Individual Agents
    1  ›  Contract Intelligence Agent
    2  ›  Risk & Benchmarking Agent
    3  ›  Policy & Compliance Agent
    4  ›  Negotiation Advisor Agent

  PART 2 — Agent Collaboration
    5  ›  Full collaborative review (all 4 agents + debate)

    C  ›  Change contract
    Q  ›  Quit
```

---

## Three-Part Demo Narrative

### Part 1 — Individual Agents (Options 1–4)

Each agent runs independently. Send the same contract to each one in turn.

**Talking points:**
- "Each agent is a specialist. Watch it decide which tools to call."
- "The LLM chooses the sequence — we didn't hardcode it."
- "Agent 1 reads and structures. Agent 2 benchmarks risk. Agent 3 checks policy. Agent 4 drafts the negotiation brief."

What to watch in each terminal:
- The `🔧` lines — every tool call the LLM decides to make
- The `→` lines — what each tool returned
- The final structured output — verdict, findings, redlines

### Part 2 — Full Collaboration (Option 5)

All four agents run in sequence. The orchestrator passes findings from each agent to the next, and agents communicate directly with each other.

**Talking points:**
- "Now watch Agent 2 push back to Agent 1 for clarification on an ambiguous clause."
- "Agent 3 is checking Agent 2's risk ratings against policy rules — watch it challenge."
- "When Agent 2 concedes, the verdict upgrades automatically."
- "Agent 4 gets the full picture from all three agents before drafting the strategy."

What to watch:
- `⟶` and `⟵` arrows — messages between agents
- `⚡ CHALLENGE:` — Agent 3 challenging Agent 2's rating
- `⚡ CHALLENGE RESOLVED` — outcome of the challenge (concede or defend)
- Terminal 5 summary at the end — Risk / Policy / Action in three lines

### Part 3 — Change Contract (Option C)

Switch to a different contract and re-run. Use this to show the same vendor appearing with a different contract, or a clean contract vs a problematic one.

---

## What Makes This Demo Compelling

**Real tool use** — the LLM genuinely decides which tools to call and in what order. The audience watches the decision process in real time.

**Real agent communication** — agents send typed messages to each other via a file-based message bus. Agent 2 can ask Agent 1 to re-extract a clause. Agent 3 can challenge Agent 2's risk rating and force a response.

**No frameworks** — built on the raw OpenAI SDK. Every line of code is readable. There is no LangChain, no AutoGen, no hidden orchestration layer.

**Genuine disagreement** — Agent 3 has its own policy rulebook and will challenge Agent 2 when their ratings conflict. The challenge and response are live, not scripted.

---

## Agent Interaction Map

```
                    ┌─────────────────┐
                    │  ORCHESTRATOR   │
                    │   (Terminal 5)  │
                    └────────┬────────┘
                             │ sends contract
                    ┌────────▼────────┐
                    │    AGENT 1      │  ◄── may receive reextract_request
                    │  Intelligence   │       from Agent 2
                    └────────┬────────┘
                             │ extraction
                    ┌────────▼────────┐
                    │    AGENT 2      │  ──► sends reextract_request to Agent 1
                    │  Risk & Bench   │  ◄── receives policy_challenge from Agent 3
                    └────────┬────────┘  ──► sends challenge_response to Agent 3
                             │ risk findings
                    ┌────────▼────────┐
                    │    AGENT 3      │  ──► sends policy_challenge to Agent 2
                    │  Policy &       │  ◄── receives challenge_response from Agent 2
                    │  Compliance     │
                    └────────┬────────┘
                             │ policy findings
                    ┌────────▼────────┐
                    │    AGENT 4      │
                    │  Negotiation    │
                    └────────┬────────┘
                             │ negotiation brief
                    ┌────────▼────────┐
                    │  ORCHESTRATOR   │
                    │  (summary)      │
                    └─────────────────┘
```

---

## Policy Rules (Agent 3)

Agent 3 checks against 10 internal policy rules:

| Rule | Area | Threshold |
|---|---|---|
| RULE_1 | Liability Cap | Minimum 6 months fees — AUTOMATIC RED below |
| RULE_2 | Payment Terms | Maximum Net 30 — AUTOMATIC RED at 60+ days |
| RULE_3 | Data Usage Rights | No vendor data usage beyond service delivery |
| RULE_4 | IP Ownership | All outputs belong to customer |
| RULE_5 | Termination Notice | Customer notice max 90 days |
| RULE_6 | Data Return | Full export within 30 days on exit |
| RULE_7 | Governing Law | India, UK, US-NY, Singapore only |
| RULE_8 | Amendment Rights | Minimum 30 days notice |
| RULE_9 | SLA | Minimum 99.5% — AUTOMATIC RED below 99.0% |
| RULE_10 | Assignment | Vendor cannot assign without consent |

---

## Risk Benchmarks (Agent 2)

| Clause | Standard | Critical Threshold |
|---|---|---|
| Liability cap | 6–12 months fees | Below 1 month |
| Payment terms | Net 30 days | 60+ days |
| SLA | 99.9% | Below 99.0% |
| Termination notice | 30–90 days | 180+ days |
| Data breach notification | 24–72 hours | 30+ days |
| Amendment notice | 30 days minimum | Below 14 days |

---

## Troubleshooting

**Agents not responding to Terminal 5**
Make sure all 4 agent terminals are running with `--service`. Check the terminal titles — each should show "Waiting for contracts..."

**`ModuleNotFoundError: No module named 'utils'`**
The agents must be run from the project root directory. `start_demo.sh` handles this automatically. If running manually, `cd` to the project root first.

**Output shows `"?"` for verdict**
Agent 3 hit the tool iteration limit. This is fixed by using standalone mode prompts (no challenge instruction) for options 1–4. In collab mode (option 5) it should not occur.

**Timeout waiting for Agent N**
One of the agent terminals may have crashed. Check each terminal for a Python traceback. The most common cause is a missing contract file path.

**`fcntl` error on Windows**
`message_bus.py` uses `fcntl` for file locking, which is Unix-only. This demo is designed for macOS or Linux.
