# 🤖 PR Review Agent

An agentic AI code reviewer that automatically reviews Python pull requests using GPT-4o-mini.
Triggered by GitHub webhooks, it runs pylint + bandit, then posts structured inline and summary
comments directly on the PR.

---

## Architecture

```
GitHub PR opened
      ↓
GitHub Webhook (POST /webhook)
      ↓
Tornado server (your laptop, port 8080)
      ↓
ngrok tunnel (exposes localhost to the internet)
      ↓
Agent Orchestrator (agent.py)
      ├── Tool: fetch PR diff        (GitHub API)
      ├── Tool: fetch PR files       (GitHub API)
      ├── Tool: fetch file content   (GitHub API)
      ├── Tool: run pylint           (local)
      ├── Tool: run bandit           (local)
      ├── Tool: post inline comment  (GitHub API)
      └── Tool: post summary comment (GitHub API)
      ↓
GPT-4o-mini decides which tools to call and synthesises the review
      ↓
Comments appear on GitHub PR ✅
```

---

## Prerequisites

- Python 3.11+
- An OpenAI API key (GPT-4o-mini access)
- A GitHub account (free) with a repo
- A GitHub Personal Access Token with `repo` scope
- ngrok installed (`brew install ngrok` on Mac)

---

## Step 1 — Clone / set up the project

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Step 2 — Configure your .env file

Copy the example and fill in your values:

```bash
cp env.example .env
```

Edit `.env`:

```
OPENAI_API_KEY=sk-...              # Your OpenAI key
GITHUB_TOKEN=ghp_...               # Your GitHub Personal Access Token
GITHUB_REPO=your-username/your-repo # e.g. john/demo-repo
GITHUB_WEBHOOK_SECRET=mydemosecret  # Any string you choose — you'll use it again in Step 5
```

### Getting your GitHub Personal Access Token
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token (classic)
3. Name: `pr-review-agent`, Expiration: 30 days
4. Check the `repo` scope only
5. Generate and copy the token immediately

---

## Step 3 — Start ngrok

In a **separate terminal**:

```bash
ngrok http 8080
```

You will see output like:
```
Forwarding   https://abc123.ngrok-free.app -> http://localhost:8080
```

**Copy the `https://` URL** — you need it for the webhook.

---

## Step 4 — Start the Tornado server

In your main terminal (with venv activated):

```bash
python server.py
```

You should see:
```
🌐 Tornado server listening on port 8080
🔗 Webhook endpoint: http://localhost:8080/webhook
💡 Health check:     http://localhost:8080/health
🤖 Model: gpt-4o-mini
```

Test the health check:
```bash
curl http://localhost:8080/health
# → {"status": "ok", "model": "gpt-4o-mini"}
```

---

## Step 5 — Configure the GitHub Webhook

1. Go to your GitHub repo → **Settings → Webhooks → Add webhook**
2. Fill in:
   - **Payload URL**: `https://abc123.ngrok-free.app/webhook` ← your ngrok URL
   - **Content type**: `application/json`
   - **Secret**: the same string you put in `GITHUB_WEBHOOK_SECRET` in your `.env`
   - **Events**: select **"Let me select individual events"** → check **Pull requests** only
3. Click **Add webhook**
4. GitHub will send a ping — you should see `200 OK` in the webhook delivery log

---

## Step 6 — Run the demo

### Option A: Use the included demo file

The `demo_code/user_api.py` file contains intentional flaws for the agent to find:
- SQL injection vulnerabilities
- Hardcoded credentials
- Insecure MD5 password hashing
- `eval()` on user input
- Shell injection via `subprocess`
- Pickle deserialisation (insecure)
- Resource leaks
- Bare `except` blocks
- Poor variable naming

To create a demo PR:
```bash
# In your GitHub repo, create a new branch and add the demo file
git checkout -b demo/flawed-user-api
cp /path/to/pr-review-agent/demo_code/user_api.py .
git add user_api.py
git commit -m "Add user API handler"
git push origin demo/flawed-user-api
```

Then open a Pull Request on GitHub from `demo/flawed-user-api` → `main`.

### Option B: Have a participant write and submit their own code

Any `.py` file opened as a PR will trigger the agent.

---

## What you will see

**In your terminal** (stdout):
```
============================================================
🚀 Agent starting review for PR #1
   Title: Add user API handler
   Author: yourname
============================================================

🧠 Agent thinking... (iteration 1)

🔧 Tool call: fetch_pr_diff({"pr_number": 1})
   → Diff fetched (2341 chars)

🔧 Tool call: fetch_pr_files({"pr_number": 1})
   → 1 file(s) changed

🔧 Tool call: fetch_file_content({"file_path": "user_api.py", "ref": "abc123..."})
   → File content fetched (1872 chars)

🧠 Agent thinking... (iteration 2)

🔧 Tool call: run_pylint({"code": "..."})
   → Pylint: Your code has been rated at 3.5/10 | 12 issue(s)

🔧 Tool call: run_bandit({"code": "..."})
   → Bandit: HIGH: 5, MEDIUM: 2, LOW: 1 | 8 issue(s)

🧠 Agent thinking... (iteration 3)

🔧 Tool call: post_inline_comment({...line 35...})
   → Inline comment posted on user_api.py:35

... (more inline comments) ...

🔧 Tool call: post_pr_comment({"pr_number": 1, "body": "## 🤖 AI Code Review..."})
   → Summary comment posted (id=123456789)

✅ Agent finished review after 5 iteration(s)
```

**On GitHub**: inline comments on specific lines + a structured summary comment.

---

## Project structure

```
pr-review-agent/
├── server.py          # Tornado webhook server
├── agent.py           # GPT-4o-mini agent + tool loop
├── config.py          # Loads settings from .env
├── tools/
│   ├── __init__.py
│   ├── github.py      # GitHub API: fetch diff, post comments
│   ├── linter.py      # Runs pylint
│   └── security.py    # Runs bandit
├── demo_code/
│   └── user_api.py    # Intentionally flawed demo file
├── requirements.txt
├── env.example
└── README.md
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Webhook returns 401 | Check `GITHUB_WEBHOOK_SECRET` matches exactly in `.env` and GitHub |
| Agent posts no comments | Check `GITHUB_TOKEN` has `repo` scope |
| `pylint` not found | Run `pip install pylint` in your venv |
| `bandit` not found | Run `pip install bandit` in your venv |
| ngrok URL expired | Free ngrok URLs reset on restart — update the webhook URL in GitHub |
| OpenAI 401 error | Check `OPENAI_API_KEY` in `.env` |

---

## Extending the demo (talking points)

- **Multi-agent**: Split into specialist agents (Security Agent, Style Agent, Logic Agent) orchestrated by a coordinator
- **Memory**: Give the agent access to past reviews so it can track recurring issues per author
- **Auto-fix**: Add a tool that opens a follow-up PR with suggested fixes applied
- **CI integration**: Block PR merges until the agent approves
- **Custom rules**: Feed the agent your team's coding standards as part of the system prompt
