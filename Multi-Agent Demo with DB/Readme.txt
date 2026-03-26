# Contract Review — Multi-Agent AI Demo (with Memory)

A multi-agent AI system that reviews vendor contracts using four autonomous specialist agents backed by persistent organisational memory. Built on plain Python, OpenAI GPT-4o, ChromaDB, and the Model Context Protocol (MCP). No frameworks. Every tool call, agent decision, inter-agent message, and memory query is visible in the terminals as it happens.

---

## What It Does

Four AI agents collaborate to review a contract, each playing a distinct role. A fifth process — the Chroma Memory MCP server — acts as the organisation's institutional memory, storing every completed review and surfacing prior history when the same vendor appears again.

| Terminal | Component | Role |
|---|---|---|
| 1 | Contract Intelligence | Reads and structures the contract. Queries memory for prior vendor history. |
| 2 | Risk & Benchmarking | Rates risk against industry benchmarks. Queries memory for similar clause precedents. Can ask Agent 1 to re-extract ambiguous clauses. |
| 3 | Policy & Compliance | Checks findings against internal policy rules. Queries memory for breach precedents. Challenges Agent 2 when ratings conflict with policy. |
| 4 | Negotiation Advisor | Drafts redlines and negotiation strategy using all prior agents' findings plus vendor negotiation history from memory. |
| 5 | Demo Controller | Menu-driven interface. Shows live memory stats in the header. |
| 6 | Chroma Memory Server | Persistent MCP server. Single process owning ChromaDB. Receives memory queries and stores completed reviews. |

---

## Why Terminal 6 Matters

Terminal 6 is the organisational memory — a named, visible service. Every agent calls it independently via the Model Context Protocol over HTTP. When Agent 4 queries vendor negotiation history, it's calling the same service Agent 1 used minutes earlier. That's how institutional knowledge scales.

The MCP server is the only process that ever touches ChromaDB, which eliminates the HNSW file-locking bug that occurs when multiple processes open a PersistentClient simultaneously.

---

## Prerequisites

**Python 3.11** is required (the `mcp` package requires 3.10+).

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install openai chromadb httpx PyPDF2 "mcp[cli]"
```

Verify everything is installed:

```bash
python -c "
import openai, chromadb, httpx, PyPDF2, mcp
from importlib.metadata import version
print('✅ All good')
print(f'  openai   : {openai.__version__}')
print(f'  chromadb : {chromadb.__version__}')
print(f'  httpx    : {httpx.__version__}')
print(f'  PyPDF2   : {PyPDF2.__version__}')
print(f'  mcp      : {version(\"mcp\")}')
"
```

---

## Project Structure

```
project-root/
  .venv311/                        ← Python 3.11 virtual environment (one level up or same dir)
  files/
    start_demo.sh                  ← One-command launcher
    demo.py                        ← Terminal 5 controller
    orchestrator.py                ← Collaborative workflow manager
    chroma_mcp_server.py           ← Terminal 6 — MCP memory server
    utils.py                       ← Shared utilities (unchanged)
    message_bus.py                 ← File-based inter-agent messaging (unchanged)
    memory.py                      ← Present but no longer imported (unchanged)
    agents/
      contract_intelligence.py     ← Agent 1
      risk_benchmarking.py         ← Agent 2
      policy_compliance.py         ← Agent 3
      negotiation_advisor.py       ← Agent 4
    sample_contract.pdf            ← DataSync Solutions Pte. Ltd. (all red flags)
    contract_B_datasync.pdf        ← Same vendor, second contract (tests memory)
    contract_C_nexahr.txt          ← Different vendor, mostly clean (AMBER outcome)
```

**Note on venv location:** `start_demo.sh` looks for `.venv311` one directory above the project files. If your venv is in the same directory as the files, edit this line in `start_demo.sh`:
```bash
VENV_ACTIVATE="$(dirname "$SCRIPT_DIR")/.venv311/bin/activate"
# Change to:
VENV_ACTIVATE="$SCRIPT_DIR/.venv311/bin/activate"
```

---

## Running the Demo

```bash
export OPENAI_API_KEY=your-key-here
chmod +x start_demo.sh
./start_demo.sh
```

This opens all 6 terminals automatically, in order:
1. Terminal 6 (Chroma Memory server) starts first with a 2-second pause
2. Terminals 1–4 (agents) start next
3. Terminal 5 (demo controller) starts last

The OpenAI API key is injected into every terminal automatically.

**Use Terminal 5 to drive the demo.**

To start completely fresh between runs:

```bash
rm -rf contract_memory_db/ messages/ memory_debug.log negotiation_briefs/
```

---

## Demo Menu (Terminal 5)

```
  CONTRACT REVIEW — MULTI-AGENT AI DEMO
  Contract : sample_contract.pdf
  Memory   : 2 contracts · 1 vendor · 8 clauses stored
  ────────────────────────────────────────────────────
  PART 1 — Individual Agents (each runs independently)
    1  ›  Contract Intelligence Agent
    2  ›  Risk & Benchmarking Agent
    3  ›  Policy & Compliance Agent
    4  ›  Negotiation Advisor Agent
  ────────────────────────────────────────────────────
  PART 2 — Agent Collaboration
    5  ›  Full collaborative review (all 4 agents + debate)
  ────────────────────────────────────────────────────
  PART 3 — Memory Advantage
    C  ›  Change contract
    Q  ›  Quit
```

Memory stats in the header update automatically after each Option 5 run.

---

## Three-Part Demo Narrative

### Part 1 — Individual Agents (Options 1–4)

Run each agent independently on `sample_contract.pdf`. Each one receives the contract, decides which tools to call, and produces a structured output.

**What to watch in each terminal:**
- `🔧` lines — every tool call the LLM decides to make
- `→` lines — what each tool returned
- `🧠` lines — memory hits (prior vendor history or similar clauses)
- The final structured output — findings, verdict, redlines

**Talking points:**
- "Each agent is a specialist. Watch it decide which tools to call — we didn't hardcode the sequence."
- "Agent 1 reads and structures. Agent 2 benchmarks risk. Agent 3 checks policy. Agent 4 drafts the negotiation brief."
- "Notice Agent 1 queries memory — it finds nothing on first run. That changes after Part 2."

### Part 2 — Full Collaboration (Option 5)

All four agents run in sequence. The orchestrator passes findings between agents. Agents communicate directly with each other via the message bus.

**What to watch:**
- `⟶` / `⟵` arrows — messages passing between agents
- Agent 2 sending `request_clarification` to Agent 1 — and Agent 1 responding
- `⚡ CHALLENGE:` — Agent 3 challenging Agent 2's risk ratings
- `⚡ CHALLENGE RESOLVED` — each challenge outcome (CONCEDED or DEFENDED)
- Terminal 6 — incoming memory calls from each agent as they query history
- Terminal 5 summary — Risk / Policy / Action verdict in three lines

**Talking points:**
- "Watch Agent 3 challenge Agent 2 — it has its own policy rulebook and will push back when ratings conflict."
- "Agent 2 must respond to every challenge. If it concedes, the verdict upgrades automatically."
- "Terminal 6 is the organisational memory. Every agent calls it independently."

**After Option 5 completes**, memory now holds the first contract review. The menu header will show `1 contract · 1 vendor · N clauses stored`.

### Part 3 — Memory Advantage (Option C then Option 5)

Press `C`, switch to `contract_B_datasync.pdf` (same vendor, different contract). Run Option 5 again.

**What to watch:**
- Agent 1 querying memory and finding the prior DataSync review — `🧠` line appears
- Agent 2 comparing current clauses against what was accepted previously
- Agent 3 citing prior breach precedents in its policy findings
- Agent 4 using prior negotiation history to sharpen opening position
- Terminal 5 summary — "HOW MEMORY CHANGED THE OUTCOME" section

**Talking points:**
- "The second time this vendor appears, every agent finds prior history."
- "Agent 4 knows what DataSync accepted before — that's the opening position."
- "This is institutional memory that doesn't leave when people leave."

---

## Agent Interaction Map

```
                    ┌─────────────────────────┐
                    │      ORCHESTRATOR        │
                    │      (Terminal 5)        │
                    └────────────┬────────────┘
                                 │ collab_request
                    ┌────────────▼────────────┐
                    │        AGENT 1           │◄── reextract_request (from Agent 2)
                    │  Contract Intelligence   │──► reextract_response (to Agent 2)
                    │     (Terminal 1)         │──► query_memory ──► Terminal 6
                    └────────────┬────────────┘
                                 │ extraction_complete
                    ┌────────────▼────────────┐
                    │        AGENT 2           │──► reextract_request (to Agent 1)
                    │   Risk & Benchmarking    │◄── policy_challenge (from Agent 3)
                    │     (Terminal 2)         │──► challenge_response (to Agent 3)
                    │                          │──► query_memory ──► Terminal 6
                    └────────────┬────────────┘
                                 │ risk_complete
                    ┌────────────▼────────────┐
                    │        AGENT 3           │──► policy_challenge (to Agent 2)
                    │  Policy & Compliance     │◄── challenge_response (from Agent 2)
                    │     (Terminal 3)         │──► query_memory ──► Terminal 6
                    └────────────┬────────────┘
                                 │ policy_complete
                    ┌────────────▼────────────┐
                    │        AGENT 4           │──► query_memory ──► Terminal 6
                    │   Negotiation Advisor    │
                    │     (Terminal 4)         │
                    └────────────┬────────────┘
                                 │ negotiation_complete
                    ┌────────────▼────────────┐
                    │      ORCHESTRATOR        │──► store_review ──► Terminal 6
                    │    (stores to memory)    │
                    └─────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   CHROMA MEMORY SERVER   │
                    │      (Terminal 6)        │
                    │  http://127.0.0.1:8765   │
                    │  Tools:                  │
                    │  • query_vendor_history  │
                    │  • query_similar_clauses │
                    │  • store_review          │
                    │  • get_stats             │
                    └─────────────────────────┘
```

---

## Memory Architecture

The memory system uses ChromaDB with OpenAI embeddings (`text-embedding-3-small`).

**Two collections:**

`contracts` — one document per completed review. Stores vendor name, verdict, liability cap, payment terms, overall risk, and a summary of all findings. Queried by agents using `query_vendor_history`.

`clauses` — individual clause values indexed by type. Stores liability cap, payment terms, data usage rights, and SLA text from each review. Queried by agents using `query_similar_clauses`.

**Two-step vendor matching:**

Exact metadata filter first (`where={"vendor": vendor_name}`). If that returns nothing — because "DataSync Solutions" ≠ "DataSync Solutions Pte. Ltd." — falls back to semantic search plus Python fuzzy matching. The fuzzy matcher strips legal suffixes (Pte, Ltd, LLC, Inc, Corp, etc.) and matches on significant words. "DataSync" matches "DataSync Solutions Pte. Ltd." reliably.

**Why a single MCP server process:**

ChromaDB's HNSW vector index is written asynchronously. When multiple processes open PersistentClient simultaneously, they compete for file locks and the index may not be readable immediately after a write. By making the MCP server the only process that touches ChromaDB, all reads and writes go through a single in-process client — no file locking, no race conditions.

---

## Policy Rules (Agent 3)

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

Agent 3 checks the highest-risk rules (RULE_1, RULE_2, RULE_3, RULE_9) in every collab run. It will challenge Agent 2 whenever a policy AUTOMATIC RED conflicts with Agent 2's risk rating.

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

## MCP Server Tools

The Chroma Memory MCP server exposes four tools over SSE at `http://127.0.0.1:8765/sse`:

**`query_vendor_history(vendor_name, n=5)`**
Semantic search for prior contracts with a named vendor. Two-step: exact metadata filter first, then semantic + fuzzy fallback. Returns list of `{document, metadata, similarity}`.

**`query_similar_clauses(clause_text, clause_type, n=5)`**
Find similar clauses across all past contracts. `clause_type` must be one of: `liability_cap`, `payment_terms`, `data_rights`, `sla`. Two-step: exact type filter first, then semantic fallback.

**`store_review(contract_id, vendor_name, contract_text, findings_json, verdict)`**
Store a completed contract review. Also stores individual clauses into the clauses collection. Called by the orchestrator after every Option 5 run.

**`get_stats()`**
Returns `{contracts: int, clauses: int, vendors: [str]}`. Used by demo.py to show memory status in the menu header.

---

## Troubleshooting

**`TypeError: FastMCP.run() got an unexpected keyword argument 'host'`**
The `host` and `port` must be on the `FastMCP()` constructor, not on `run()`. Check `chroma_mcp_server.py` contains:
```python
mcp = FastMCP("chroma-memory", host=MCP_HOST, port=MCP_PORT)
...
mcp.run(transport="sse")
```

**`Memory unavailable: unhandled errors in a TaskGroup`**
The agent can't reach the MCP server. Check Terminal 6 is running. Then verify the URL in each agent is `http://127.0.0.1:8765/sse` (not `localhost`). Test connectivity with:
```bash
python -c "
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client
async def test():
    async with sse_client(url='http://127.0.0.1:8765/sse') as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            result = await s.call_tool('get_stats', {})
            print(result.content[0].text)
asyncio.run(test())
"
```

**Agent 3 verdict shows `"?"`**
Agent 3 hit the tool iteration limit (10 calls) before producing its final JSON. Check `policy_compliance.py` has two separate prompts — `SYSTEM_PROMPT` (collab) and `SYSTEM_PROMPT_STANDALONE` — and that the collab prompt caps `check_policy_rule` at 4 calls max.

**Only first challenge response processed**
Check that the challenge handling loop in `policy_compliance.py` iterates over all challenges sent, not just waits once. The loop should be `for _ in _state["challenges_sent"]:`.

**`ModuleNotFoundError: No module named 'utils'`**
Agents must run from the project root. `start_demo.sh` handles this via `cd '$SCRIPT_DIR'`. If running manually, `cd` to the project root first.

**`Virtual environment not found at .venv311/`**
The script looks one directory above the project files. Either move `.venv311` to the parent directory, or edit `start_demo.sh` to change `VENV_ACTIVATE` to point to your venv's actual location.

**Memory not persisting between runs**
Do not delete `contract_memory_db/` between runs if you want memory to persist. Only delete it when you want a completely fresh start.

**`fcntl` error**
`message_bus.py` uses `fcntl` for file locking, which is Unix-only. This demo runs on macOS or Linux only.

---

## Setup Checklist

- [ ] Python 3.11 available (`python3.11 --version`)
- [ ] `.venv311` created and packages installed
- [ ] `OPENAI_API_KEY` exported in shell
- [ ] `start_demo.sh` in same directory as `demo.py` and agent files
- [ ] `.venv311` location matches `VENV_ACTIVATE` path in `start_demo.sh`
- [ ] `chmod +x start_demo.sh` run
- [ ] Sample contracts in project root (`sample_contract.pdf`, `contract_B_datasync.pdf`, `contract_C_nexahr.txt`)
- [ ] Connectivity test passes (see Troubleshooting above)

---

## Stack

```
Language  : Python 3.11
LLM       : OpenAI GPT-4o
Embeddings: OpenAI text-embedding-3-small
Memory    : ChromaDB (local persistent)
Protocol  : Model Context Protocol (MCP) 1.26.0 — SSE transport
Messaging : File-based JSON message bus
Frameworks: None
```
