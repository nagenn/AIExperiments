"""
agent.py — GPT-4o-mini powered PR review agent.

The agent is given a set of tools and autonomously decides:
  1. Read context.txt from the repo root (if it exists) to understand the project
  2. Which tools to call (fetch diff, lint, security scan)
  3. In what order
  4. How to synthesize findings into a structured review
  5. Where to post comments (inline vs summary)
"""

import json
import textwrap
import time
from openai import OpenAI

import config
from tools.github import (
    fetch_pr_diff,
    fetch_pr_files,
    fetch_file_content,
    post_pr_comment,
    post_inline_comment,
)
from tools.linter import run_pylint
from tools.security import run_bandit

client = OpenAI(api_key=config.OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling schema)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_context_file",
            "description": (
                "Fetch the context.txt file from the root of the repository. "
                "Always call this FIRST before anything else. "
                "It contains a description of the project, its stack, file purposes, "
                "and specific review focus areas. "
                "If the file does not exist, proceed without it — do not stop."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_pr_diff",
            "description": "Fetch the full unified diff for the pull request. Use this to understand what changed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pr_number": {"type": "integer", "description": "The pull request number."}
                },
                "required": ["pr_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_pr_files",
            "description": "Fetch the list of files changed in the PR along with their patch and raw URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pr_number": {"type": "integer", "description": "The pull request number."}
                },
                "required": ["pr_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_file_content",
            "description": (
                "Fetch the full content of a specific file at the PR's head commit. "
                "Use this to get the complete file for linting and security scanning."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file in the repo."},
                    "ref": {"type": "string", "description": "The git ref (commit SHA or branch) to fetch from."},
                },
                "required": ["file_path", "ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_pylint",
            "description": "Run pylint on a Python code string. Returns style issues, code smells, and a quality score.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The Python source code to lint."},
                    "filename": {"type": "string", "description": "Filename hint for pylint output."},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bandit",
            "description": "Run bandit security scanner on a Python code string. Returns security vulnerabilities by severity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The Python source code to scan."},
                    "filename": {"type": "string", "description": "Filename hint for bandit output."},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "post_inline_comment",
            "description": "Post an inline review comment on a specific line of a file in the PR. Use for precise, line-level feedback.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pr_number": {"type": "integer"},
                    "commit_sha": {"type": "string", "description": "The head commit SHA of the PR."},
                    "path": {"type": "string", "description": "File path relative to repo root."},
                    "line": {"type": "integer", "description": "Line number to comment on."},
                    "body": {"type": "string", "description": "The comment text."},
                },
                "required": ["pr_number", "commit_sha", "path", "line", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "post_pr_comment",
            "description": "Post a general summary comment on the PR. Use this for the final structured review summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pr_number": {"type": "integer"},
                    "body": {"type": "string", "description": "The markdown-formatted summary review."},
                },
                "required": ["pr_number", "body"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Pretty printer helpers
# ---------------------------------------------------------------------------

W = 70  # terminal width

def hr(char="─"):
    print(char * W)

def section(title):
    print(f"\n{'─'*W}")
    print(f"  {title}")
    print(f"{'─'*W}")

def row(label, value, width=22):
    print(f"  {label:<{width}} {value}")

def print_tool_table(tool_log: list):
    """Print a clean summary table of all tool calls at the end."""
    print(f"\n{'═'*W}")
    print(f"  {'TOOL CALL SUMMARY':^{W-2}}")
    print(f"{'═'*W}")
    print(f"  {'#':<4} {'TOOL':<30} {'RESULT':<34}")
    print(f"  {'─'*4} {'─'*30} {'─'*34}")
    for i, entry in enumerate(tool_log, 1):
        tool  = entry["tool"][:29]
        result = entry["result"][:33]
        print(f"  {i:<4} {tool:<30} {result:<34}")
    print(f"{'═'*W}")

def print_findings_table(findings: list):
    """Print a clean table of all lint/security findings with line numbers."""
    if not findings:
        print("  ✅  No findings recorded.")
        return
    print(f"\n  {'FILE':<22} {'LINE':<6} {'TYPE':<10} {'SEVERITY':<10} {'CODE / MESSAGE'}")
    print(f"  {'─'*22} {'─'*6} {'─'*10} {'─'*10} {'─'*28}")
    for f in findings:
        file     = f["file"][:21]
        line     = str(f.get("line", "─"))[:5]
        ftype    = f["type"][:9]
        severity = f["severity"][:9]
        detail   = f["detail"][:40]
        print(f"  {file:<22} {line:<6} {ftype:<10} {severity:<10} {detail}")

def print_legend():
    """Print a legend explaining scores and severity levels."""
    print(f"\n{'─'*W}")
    print(f"  📖  LEGEND")
    print(f"{'─'*W}")
    print(f"  Pylint score    10/10 = perfect  |  7+/10 = good  |  <5/10 = needs work")
    print(f"  Bandit HIGH   — Serious security vulnerability. Fix before merge.")
    print(f"  Bandit MEDIUM — Potential security risk. Review carefully.")
    print(f"  Bandit LOW    — Minor security note. Use your judgement.")
    print(f"  Style         — Code quality, PEP8, or complexity issue.")
    print(f"{'─'*W}")


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def dispatch_tool(name: str, args: dict, pr_number: int, pr_meta: dict,
                  tool_log: list, findings: list) -> str:
    """Execute a tool call, log it, and return result string for the agent."""

    # ── fetch_context_file ──────────────────────────────────────────────────
    if name == "fetch_context_file":
        content = fetch_file_content(config.GITHUB_REPO, "context.txt", "main")
        if not content:
            msg = "context.txt not found — proceeding without project context."
            tool_log.append({"tool": "fetch_context_file", "result": "⚠️  Not found"})
            return msg
        tool_log.append({"tool": "fetch_context_file", "result": "✅ Loaded"})
        return content

    # ── fetch_pr_diff ───────────────────────────────────────────────────────
    elif name == "fetch_pr_diff":
        result = fetch_pr_diff(args["pr_number"])
        tool_log.append({"tool": "fetch_pr_diff", "result": "✅ Done"})
        return result[:6000]

    # ── fetch_pr_files ──────────────────────────────────────────────────────
    elif name == "fetch_pr_files":
        files = fetch_pr_files(args["pr_number"])
        simplified = [
            {
                "filename": f["filename"],
                "status": f["status"],
                "additions": f["additions"],
                "deletions": f["deletions"],
                "patch": f.get("patch", "")[:1000],
            }
            for f in files
        ]
        names = ", ".join(f["filename"] for f in files)
        tool_log.append({"tool": "fetch_pr_files", "result": f"{len(files)} file(s)"})
        row("📁 Files in PR", names)
        return json.dumps(simplified)

    # ── fetch_file_content ──────────────────────────────────────────────────
    elif name == "fetch_file_content":
        content = fetch_file_content(config.GITHUB_REPO, args["file_path"], args["ref"])
        tool_log.append({"tool": f"fetch_file({args['file_path']})", "result": "✅ Done"})
        row("📥 Reading", args["file_path"])
        return content[:5000]

    # ── run_pylint ──────────────────────────────────────────────────────────
    elif name == "run_pylint":
        result = run_pylint(args["code"], args.get("filename", "review_target.py"))
        fname = args.get("filename", "?")
        issues = result["issues"]
        # Extract just the score number e.g. "7.50/10"
        score_str = result["score"]
        score_num = score_str.split("rated at ")[-1].split(" ")[0] if "rated at" in score_str else "?"
        tool_log.append({"tool": f"pylint({fname})", "result": f"Score {score_num}  |  {len(issues)} issue(s)"})
        row(f"🔍 pylint  {fname}", f"Score {score_num}  |  {len(issues)} issue(s)")
        for issue in issues:
            # Parse pylint line e.g. "/tmp/xxx/app.py:24:0: W0611: Unused import os (unused-import)"
            # Extract: line number, code, message
            parts = issue.strip().split(":")
            line_num = parts[1].strip() if len(parts) > 1 else "─"
            # Get the message part after the line/col info
            rest = ":".join(parts[3:]).strip() if len(parts) > 3 else issue.strip()
            # Trim to "W0611 unused-import" style
            rest_parts = rest.split("(")
            code_msg = rest_parts[-1].rstrip(")") if "(" in rest else rest
            findings.append({
                "file": fname,
                "type": "pylint",
                "severity": "style",
                "line": line_num,
                "detail": code_msg.strip()[:45],
            })
        return json.dumps(result)

    # ── run_bandit ──────────────────────────────────────────────────────────
    elif name == "run_bandit":
        result = run_bandit(args["code"], args.get("filename", "review_target.py"))
        fname = args.get("filename", "?")
        issues = result["issues"]
        high   = sum(1 for i in issues if i.get("severity") == "HIGH")
        medium = sum(1 for i in issues if i.get("severity") == "MEDIUM")
        low    = sum(1 for i in issues if i.get("severity") == "LOW")
        summary = f"HIGH: {high}  MEDIUM: {medium}  LOW: {low}"
        tool_log.append({"tool": f"bandit({fname})", "result": summary})
        row(f"🔒 bandit  {fname}", summary)
        for issue in issues:
            findings.append({
                "file": fname,
                "type": "security",
                "severity": issue.get("severity", "?"),
                "line": str(issue.get("line", "─")),
                "detail": issue.get("description", "")[:45],
            })
        return json.dumps(result)

    # ── post_inline_comment ─────────────────────────────────────────────────
    elif name == "post_inline_comment":
        result = post_inline_comment(
            args["pr_number"],
            args["commit_sha"],
            args["path"],
            args["line"],
            args["body"],
        )
        label = f"{args['path']}  line {args['line']}"
        tool_log.append({"tool": "inline_comment", "result": label})
        row("💬 Inline comment →", f"{args['path']}  line {args['line']}")
        return json.dumps({"status": "posted", "id": result.get("id")})

    # ── post_pr_comment ─────────────────────────────────────────────────────
    elif name == "post_pr_comment":
        result = post_pr_comment(args["pr_number"], args["body"])
        tool_log.append({"tool": "post_pr_comment", "result": "✅ Posted"})
        row("📝 Summary →", f"Posted to GitHub PR #{args['pr_number']}")
        return json.dumps({"status": "posted", "id": result.get("id")})

    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Python code reviewer acting as an autonomous agent.
    You have been triggered by a new pull request and must perform a thorough review.

    Your review process — follow this order strictly:

    1. ALWAYS call fetch_context_file first. This gives you the project description,
       stack, file purposes, and specific areas to focus on. If it returns a
       "not found" message, proceed without it — do not stop or ask questions.

    2. Fetch the PR diff and file list to understand what changed.

    3. For each Python file changed, fetch its full content.

    4. Run pylint on each Python file to find style, complexity, and quality issues.

    5. Run bandit on each Python file to find security vulnerabilities.

    6. Post INLINE comments on specific lines for precise, actionable issues.
       Be surgical — only comment on lines that genuinely warrant it.

    7. Finally, post ONE SUMMARY comment on the PR with this exact structure:

    ---

    ## 🤖 AI Code Review Summary

    ### 🐛 Bugs & Logic Issues
    <findings or "None found">

    ### 🔒 Security
    <bandit findings with severity — be specific about what the vulnerability is and how to fix it>

    ### 🎨 Code Quality & Style
    <pylint findings and score — highlight the most important ones>

    ### ✅ What looks good
    <genuine positive observations — don't skip this section>

    ### 📋 Recommendation
    **APPROVE** / **REQUEST CHANGES** / **NEEDS DISCUSSION**
    <one sentence explaining your recommendation>

    ---

    General guidance:
    - Be specific and constructive — reference file names and line numbers
    - Tailor your review to the project context from context.txt when available
    - Use the project's stack knowledge to give relevant, targeted advice
    - Prioritise HIGH severity issues over minor style nits
    - Keep inline comments concise — one issue per comment
""")


def run_agent(pr_number: int, pr_meta: dict):
    """
    Main agent loop. Runs until the agent decides it's done
    (no more tool calls).
    """
    start_time = time.time()
    tool_log   = []   # track every tool call
    findings   = []   # track all lint/security findings

    # ── Header ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print(f"  🚀  AI CODE REVIEW AGENT  —  PR #{pr_number}")
    print(f"{'═'*W}")
    row("Title",  pr_meta.get("title", "N/A"))
    row("Author", pr_meta.get("user", {}).get("login", "N/A"))
    row("Branch", f"{pr_meta.get('head', {}).get('ref', '?')} → {pr_meta.get('base', {}).get('ref', '?')}")
    row("Model",  config.OPENAI_MODEL)
    print(f"{'─'*W}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Please review pull request #{pr_number}.\n"
                f"Title: {pr_meta.get('title')}\n"
                f"Author: {pr_meta.get('user', {}).get('login')}\n"
                f"Head commit SHA: {pr_meta.get('head', {}).get('sha')}\n"
                f"Base branch: {pr_meta.get('base', {}).get('ref')}\n"
                f"Head branch: {pr_meta.get('head', {}).get('ref')}\n\n"
                f"Start by fetching context.txt, then proceed with your review."
            ),
        },
    ]

    iteration = 0
    max_iterations = 20

    while iteration < max_iterations:
        iteration += 1

        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message
        messages.append(message)

        # Agent is done — no more tool calls
        if not message.tool_calls:
            elapsed = time.time() - start_time

            # ── Findings table ────────────────────────────────────────────
            section("📋  RAW TOOL FINDINGS  (input to AI review)")
            print_findings_table(findings)
            print_legend()

            # ── Tool call summary table ───────────────────────────────────
            section(f"⚙️   TOOL CALL LOG  ({len(tool_log)} calls  |  {iteration} iterations  |  {elapsed:.1f}s)")
            print_tool_table(tool_log)

            # ── Done ──────────────────────────────────────────────────────
            print(f"\n  ✅  Review complete — check GitHub PR #{pr_number} for comments.")
            print(f"{'═'*W}\n")
            break

        # Process each tool call
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            tool_result = dispatch_tool(
                tool_name, tool_args, pr_number, pr_meta, tool_log, findings
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })

    if iteration >= max_iterations:
        print(f"\n  ⚠️  Max iterations ({max_iterations}) reached — agent stopped.")
        print(f"{'═'*W}\n")
