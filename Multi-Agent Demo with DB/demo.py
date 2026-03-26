"""
demo.py — THE DEMO CONTROLLER (Terminal 5)

Three-part demo:
  PART 1 — Individual agents   (options 1-4)
  PART 2 — Collaboration       (option 5)
  PART 3 — Memory advantage    (C to change contract and re-run)
"""

import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import read_contract, prompt_contract_file
from message_bus import get_messages, mark_read, send_message
import orchestrator

BAR  = "═" * 60
BAR2 = "─" * 60

AGENT_NAMES = {
    "1": ("contract_intelligence", "Contract Intelligence Agent",
          "read_contract | query_memory | identify_clauses | "
          "flag_ambiguity | summarise_plain_english"),
    "2": ("risk_benchmarking",     "Risk & Benchmarking Agent",
          "read_contract | query_memory | benchmark_clause | "
          "request_clarification | flag_risk"),
    "3": ("policy_compliance",     "Policy & Compliance Agent",
          "read_contract | query_memory | check_policy_rule | "
          "challenge_risk_agent | issue_verdict"),
    "4": ("negotiation_advisor",   "Negotiation Advisor Agent",
          "read_contract | query_memory | draft_redline | "
          "build_strategy | issue_negotiation_brief"),
}


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


def _get_stats():
    """Fetch memory stats from MCP server. Returns safe defaults on failure."""
    try:
        return json.loads(_call_mcp("get_stats", {}))
    except Exception:
        return {"contracts": 0, "vendors": [], "clauses": 0}


# ── Display helpers ───────────────────────────────────────────────────────────

def log(t):
    print(f"  {t}")

def section(t):
    print(f"\n  ── {t}")
    print(f"  {BAR2[:50]}")


def print_menu(contract_file):
    stats = _get_stats()

    print(f"\n  {BAR}")
    print(f"  CONTRACT REVIEW — MULTI-AGENT AI DEMO")
    print(f"  {BAR}")
    print(f"  Contract : {os.path.basename(contract_file)}")
    mem_status = (
        f"{stats['contracts']} contracts · "
        f"{len(stats['vendors'])} vendors · "
        f"{stats['clauses']} clauses stored"
    )
    print(f"  Memory   : {mem_status}")
    print(f"  {BAR2}")
    print(f"  PART 1 — Individual Agents (each runs independently)")
    print(f"    1  ›  Contract Intelligence Agent")
    print(f"           Tools: read_contract | query_memory | identify_clauses ...")
    print(f"    2  ›  Risk & Benchmarking Agent")
    print(f"           Tools: read_contract | query_memory | benchmark_clause ...")
    print(f"    3  ›  Policy & Compliance Agent")
    print(f"           Tools: read_contract | query_memory | check_policy_rule ...")
    print(f"    4  ›  Negotiation Advisor Agent")
    print(f"           Tools: read_contract | query_memory | draft_redline ...")
    print(f"  {BAR2}")
    print(f"  PART 2 — Agent Collaboration")
    print(f"    5  ›  Full collaborative review (all 4 agents + debate)")
    print(f"  {BAR2}")
    print(f"  PART 3 — Memory Advantage")
    print(f"    C  ›  Change contract  (switch and re-run — watch memory surface)")
    print(f"  {BAR2}")
    print(f"    Q  ›  Quit")
    print(f"  {BAR}\n")


def wait_for_agent(agent_id, agent_label, choice, timeout=300):
    start = time.time()
    print(f"\n  ⏳ Waiting for Terminal {choice} ({agent_label})",
          end="", flush=True)
    while time.time() - start < timeout:
        msgs = get_messages("controller", message_type="standalone_result")
        for fp, msg in msgs:
            if msg["content"].get("agent") == agent_id:
                mark_read(fp)
                print(" ✅")
                return msg
        print(".", end="", flush=True)
        time.sleep(2)
    print(f"\n  ⚠️  Timeout — is Terminal {choice} running with --service?")
    return None


def run_individual(choice, contract_file, contract_text):
    agent_id, agent_label, tools = AGENT_NAMES[choice]

    print(f"\n  {BAR}")
    print(f"  PART 1 — {agent_label}")
    print(f"  {BAR}")
    log(f"Tools: {tools}")
    print()
    log(f"Sending contract to Terminal {choice}...")
    log(f"Watch Terminal {choice} — the LLM will decide which tools to call.\n")

    send_message("controller", agent_id, "standalone_request", {
        "contract_file": contract_file,
        "contract_text": contract_text
    })

    msg = wait_for_agent(agent_id, agent_label, choice)
    if not msg:
        return

    summary = msg["content"].get("summary", "")
    section(f"SUMMARY — {agent_label.upper()}")
    print(summary)
    print()
    log(f"Full output visible in Terminal {choice}.")


def run_collaboration(contract_file, contract_text):
    print(f"\n  {BAR}")
    print(f"  PART 2 — FULL COLLABORATIVE REVIEW")
    print(f"  {BAR}")
    log("All 4 terminals will light up in sequence.")
    log("Watch for:")
    log("  • Each agent calling query_memory — what does it find?")
    log("  • Agent 2 sending request_clarification → Agent 1 responds")
    log("  • Agent 3 sending challenge_risk_agent → Agent 2 responds")
    log("  • Agent 4 using memory to shape negotiation strategy")
    print()
    log("Starting in 3 seconds...")
    time.sleep(3)

    result = orchestrator.run(contract_file, contract_text)

    section("COLLABORATIVE REVIEW — CONTROLLER SUMMARY")

    risk   = result.get("risk_findings", {})
    policy = result.get("policy_findings", {})
    neg    = result.get("negotiation", {})

    overall = risk.get("overall_risk", "?")
    pv      = policy.get("overall_verdict", "?")
    nv      = neg.get("negotiation_verdict", "?")

    ricon = {"RED": "🔴", "AMBER": "🟡", "GREEN": "🟢"}.get(overall, "⚪")
    picon = {"APPROVED": "✅", "CONDITIONAL": "🟡", "REJECTED": "🔴"}.get(pv, "⚪")
    nicon = {"NEGOTIATE": "🟡", "WALK_AWAY": "🔴",
             "ACCEPT_WITH_CONDITIONS": "🟢"}.get(nv, "⚪")

    log(f"  • Vendor  : {result.get('vendor', '?')}")
    log(f"  • Risk    : {ricon}  {overall}")
    log(f"  • Policy  : {picon}  {pv}")
    log(f"  • Action  : {nicon}  {nv.replace('_', ' ')}")

    mem_contributions = []
    if risk.get("memory_influenced_assessment"):
        mem_contributions.append(
            f"Agent 2: {risk['memory_influenced_assessment'][:90]}")
    if policy.get("memory_influenced_verdict"):
        mem_contributions.append(
            f"Agent 3: {policy['memory_influenced_verdict'][:90]}")
    if neg.get("memory_advantages"):
        mem_contributions.append(
            f"Agent 4: {len(neg['memory_advantages'])} memory advantage(s) applied")

    if mem_contributions:
        section("🧠 HOW MEMORY CHANGED THE OUTCOME")
        for m in mem_contributions:
            log(f"  • {m}")
    else:
        log("\n  🧠 No prior history found — this review is now in memory.")
        log("     Press C to switch contracts and watch memory surface next run.")

    breaches  = policy.get("policy_breaches", [])
    auto_reds = [b for b in breaches if "RED" in b.get("severity", "")]
    if auto_reds:
        log(f"\n  🔴  Automatic policy breaches: {len(auto_reds)}")
    redlines = neg.get("redline_suggestions", [])
    if redlines:
        log(f"  ✏️   Redlines drafted: {len(redlines)} clauses")

    stats = _get_stats()
    log(f"\n  🧠  Memory: {stats['contracts']} contract(s) from "
        f"{len(stats['vendors'])} vendor(s) on record.")

    print()
    log("Full detail visible in agent terminals.")


def change_contract():
    print(f"\n  {BAR}")
    print(f"  PART 3 — MEMORY ADVANTAGE / CHANGE CONTRACT")
    print(f"  {BAR}")

    stats = _get_stats()
    if stats["contracts"] == 0:
        log("  No contracts in memory yet.")
        log("  Run option 5 first, then switch contracts to see memory work.")
    else:
        log(f"  Memory holds:")
        log(f"    • {stats['contracts']} contract(s) reviewed")
        log(f"    • {len(stats['vendors'])} vendor(s): "
            f"{', '.join(stats['vendors'])}")
        log(f"    • {stats['clauses']} clause(s) indexed")
        print()
        log("  On the next run the agents will:")
        log("    • Call query_memory and find this history")
        log("    • Surface prior verdicts for matching vendors")
        log("    • Benchmark clauses against what you have accepted before")
        log("    • Use vendor history to sharpen negotiation strategy")

    print()
    new_file = prompt_contract_file()
    if not new_file:
        return None, None
    new_text = read_contract(new_file)
    log(f"\n  Switched to: {os.path.basename(new_file)}")
    return new_file, new_text


def main():
    if len(sys.argv) > 1:
        contract_file = sys.argv[1]
        if not os.path.exists(contract_file):
            print(f"\n  ❌  File not found: {contract_file}\n")
            sys.exit(1)
    else:
        contract_file = prompt_contract_file()

    contract_text = read_contract(contract_file)
    log(f"\n  Contract loaded: {os.path.basename(contract_file)}\n")

    while True:
        print_menu(contract_file)
        choice = input("  Enter choice: ").strip().upper()

        if choice in ("1", "2", "3", "4"):
            run_individual(choice, contract_file, contract_text)
            input("\n  Press Enter to return to menu...")

        elif choice == "5":
            run_collaboration(contract_file, contract_text)
            input("\n  Press Enter to return to menu...")

        elif choice == "C":
            new_file, new_text = change_contract()
            if new_file:
                contract_file = new_file
                contract_text = new_text

        elif choice in ("Q", "QUIT", "EXIT"):
            print("\n  Goodbye.\n")
            sys.exit(0)

        else:
            print("  Invalid choice. Enter 1-5, C, or Q.")


if __name__ == "__main__":
    main()
