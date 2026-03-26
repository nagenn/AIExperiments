"""
utils.py
--------
Shared utilities — PDF reading, display helpers, OpenAI client,
robust JSON parsing, and the tool-use loop.
"""

import os
import sys
import json
import re
from openai import OpenAI

MODEL  = "gpt-4o"
SILENT = "__SILENT__"


# ── OpenAI client ─────────────────────────────────────────────────────────────

def get_client():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("\n  ❌  OPENAI_API_KEY not set.")
        print("      Run: export OPENAI_API_KEY=your-key-here\n")
        sys.exit(1)
    return OpenAI(api_key=key)


# ── Contract reading ──────────────────────────────────────────────────────────

def read_contract(filepath):
    """Read a contract from .txt or .pdf — returns plain text either way."""
    if not os.path.exists(filepath):
        print(f"\n  ❌  File not found: {filepath}\n")
        sys.exit(1)

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        try:
            import PyPDF2
        except ImportError:
            print("\n  ❌  PDF support requires PyPDF2.")
            print("      Run: pip install pypdf2\n")
            sys.exit(1)
        text = ""
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        if not text.strip():
            print("\n  ❌  PDF text extraction returned empty content.")
            print("      This PDF may be scanned. Open in Word and re-export.\n")
            sys.exit(1)
        return text
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()


def prompt_contract_file():
    """Show available contracts and prompt for a filename."""
    print()
    contracts = sorted([
        f for f in os.listdir(".")
        if f.endswith((".txt", ".pdf")) and not f.startswith(".")
    ])
    if contracts:
        print("  Available contracts:")
        for c in contracts:
            print(f"    • {c}")
        print()
    filepath = input("  Enter contract file path: ").strip()
    if not filepath:
        print("  No file provided. Exiting.")
        sys.exit(1)
    return filepath


# ── JSON parsing ──────────────────────────────────────────────────────────────

def parse_llm_json(content):
    """
    Robustly parse JSON from LLM output.
    Handles all common formats:
      1. Clean JSON
      2. ```json ... ``` fenced
      3. Preamble sentence + fenced JSON
      4. Preamble + bare JSON block
    """
    content = content.strip()

    # Case 1: clean JSON
    try:
        return json.loads(content)
    except Exception:
        pass

    # Case 2 & 3: markdown fences
    fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except Exception:
            pass

    # Case 4: outermost { } block
    brace_match = re.search(r'\{[\s\S]*\}', content)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except Exception:
            pass

    return {"raw_output": content}


def unwrap_result(result):
    """
    Safety net — if a result is {"raw_output": "..."}, parse it.
    Call this at the top of any function that consumes agent results.
    """
    if not isinstance(result, dict):
        return {}
    if "raw_output" not in result:
        return result
    return parse_llm_json(result["raw_output"])


# ── Display helpers ───────────────────────────────────────────────────────────

BAR  = "═" * 60
BAR2 = "─" * 60


def header(title, subtitle=""):
    print(f"\n{BAR}")
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(BAR)


def section(text):
    print(f"\n  ── {text}")
    print(f"  {BAR2[:50]}")


def log(text):
    print(f"  {text}")


def thinking(text):
    # Suppressed in clean mode
    pass


def tool_call(name, params):
    """Show tool name only — no parameters."""
    print(f"  🔧 {name}")


def tool_result(content):
    """Show a trimmed one-line result."""
    trimmed = content.replace("\n", " ").strip()
    if len(trimmed) > 100:
        trimmed = trimmed[:97] + "..."
    print(f"     → {trimmed}")


def tool_no_result(content):
    trimmed = content.replace("\n", " ").strip()[:100]
    print(f"     → {trimmed}")


def memory_hit(content):
    trimmed = content.replace("\n", " ").strip()
    if len(trimmed) > 110:
        trimmed = trimmed[:107] + "..."
    print(f"  🧠 {trimmed}")


def sending(to, text):
    print(f"\n  ⟶  {to.upper()}: {text[:80]}")


def receiving(frm, text):
    print(f"\n  ⟵  {frm.upper()}: {text[:80]}")


def challenge_log(text):
    print(f"\n  ⚡ CHALLENGE: {text[:100]}")


def resolved_log(text):
    print(f"  ✅ RESOLVED:  {text[:100]}")


def risk_line(label, level, finding):
    icons = {"CRITICAL": "🔴", "HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    icon  = icons.get(level.upper(), "⚪")
    finding_short = finding[:80] if finding else ""
    print(f"  {icon} {label.upper():20s} [{level}]  {finding_short}")


# ── Tool-use loop ─────────────────────────────────────────────────────────────

def run_tool_loop(client, system_prompt, user_prompt, tools, tool_handler):
    """
    Core agentic loop. LLM decides which tools to call.
    Loops until LLM produces a final answer with no tool calls.
    Returns a proper dict — never raw_output if avoidable.

    If a tool handler returns SILENT, the tool call and result
    are completely suppressed from terminal output.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt}
    ]

    max_iterations = 10
    iteration      = 0

    while iteration < max_iterations:
        iteration += 1

        response = client.chat.completions.create(
            model       = MODEL,
            messages    = messages,
            tools       = tools,
            tool_choice = "auto",
            temperature = 0.1,
        )

        msg = response.choices[0].message

        # ── Agent chose to call tools ─────────────────────────────────────────
        if msg.tool_calls:
            messages.append(msg)

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}

                result_str = tool_handler(name, args)

                # Only print if not silenced
                if result_str != SILENT:
                    tool_call(name, args)
                    tool_result(result_str)

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result_str if result_str != SILENT else "OK."
                })

        # ── Final answer ──────────────────────────────────────────────────────
        else:
            content = msg.content or ""
            return parse_llm_json(content)

    return {"error": "Max iterations reached without final answer"}