═══════════════════════════════════════════════════════════════════════
  AI PR REVIEW AGENT — DEMO RUNBOOK
  Books Manager Project
═══════════════════════════════════════════════════════════════════════

OVERVIEW
────────
An agentic AI that automatically reviews Python pull requests.
When a PR is opened on GitHub, the agent:
  1. Reads context.txt to understand the project
  2. Fetches the PR diff and changed files
  3. Runs pylint (code quality) on each Python file
  4. Runs bandit (security scanner) on each Python file
  5. Posts inline comments on specific lines
  6. Posts a structured summary review on the PR

Stack: Python, Tornado, GPT-4o-mini, GitHub Webhooks, ngrok


PRE-REQUISITES (one time setup)
────────────────────────────────
  - Python 3.9+
  - ngrok installed (brew install ngrok)
  - ngrok account + authtoken configured (ngrok config add-authtoken <token>)
  - OpenAI API key
  - GitHub Personal Access Token (fine-grained, BooksManager repo only)
    Permissions needed:
      - Contents       → Read-only
      - Pull requests  → Read and Write
      - Metadata       → Read-only
  - pr-review-agent project folder set up with:
      pip install -r requirements.txt


PROJECT FOLDER STRUCTURE
─────────────────────────
  pr-review-agent/
  ├── server.py           Tornado webhook server
  ├── agent.py            GPT-4o-mini agent + tool loop
  ├── config.py           Loads settings from .env
  ├── setup_env.sh        Sets environment variables
  ├── tools/
  │   ├── github.py       Fetch PR, post comments
  │   ├── linter.py       Runs pylint
  │   └── security.py     Runs bandit
  └── context.txt         ← also lives in BooksManager repo root


STEP 1 — SET ENVIRONMENT VARIABLES
─────────────────────────────────────
In BOTH terminals run:

  source setup_env.sh

Verify in each terminal:

  echo $OPENAI_API_KEY       ← should print your key
  echo $GITHUB_TOKEN         ← should print your token
  echo $GITHUB_REPO          ← should print nagenn/BooksManager
  echo $GITHUB_WEBHOOK_SECRET ← should print your secret


STEP 2 — START THE TORNADO SERVER
────────────────────────────────────
In Terminal 1:

  cd pr-review-agent
  source venv/bin/activate
  python server.py

Expected output:
  🌐 Tornado server listening on port 8080
  🔗 Webhook endpoint: http://localhost:8080/webhook
  💡 Health check:     http://localhost:8080/health
  🤖 Model: gpt-4o-mini


STEP 3 — START NGROK
──────────────────────
In Terminal 2:

  source setup_env.sh
  ngrok http 8080

Copy the https:// forwarding URL e.g.:
  https://abc123.ngrok-free.app


STEP 4 — VERIFY GITHUB WEBHOOK
─────────────────────────────────
Go to: BooksManager repo → Settings → Webhooks

Check:
  - Payload URL   : https://abc123.ngrok-free.app/webhook
  - Content type  : application/json   ← IMPORTANT
  - Secret        : matches GITHUB_WEBHOOK_SECRET in setup_env.sh
  - Events        : Pull requests only

NOTE: ngrok URL changes every time you restart ngrok (free tier).
      Update the webhook payload URL each session.

To update webhook URL:
  GitHub → Settings → Webhooks → Edit → update Payload URL → Save


STEP 5 — TRIGGER THE DEMO
───────────────────────────
Option A — via GitHub UI (recommended for demo):
  1. Go to BooksManager repo on GitHub
  2. Click branch dropdown → type new branch name → Create branch
  3. Navigate to any .py file (e.g. handlers.py)
  4. Click pencil icon (Edit)
  5. Make a small change (add a comment line)
  6. Commit changes to the new branch
  7. Click "Compare & pull request" banner
  8. Click "Create pull request"

Option B — via terminal:
  git checkout -b demo/test-review
  # edit any .py file
  git add .
  git commit -m "Test PR for agent demo"
  git push origin demo/test-review
  Then open PR on GitHub.


WHAT TO WATCH
──────────────
Terminal 1 shows the agent working in real time:

  ══════════════════════════════════════════════════════════════════════
    🚀  AI CODE REVIEW AGENT  —  PR #1
  ══════════════════════════════════════════════════════════════════════
    Title                  your PR title
    Author                 nagenn
    Branch                 demo/test-review → main
  ──────────────────────────────────────────────────────────────────────
    📁 Files in PR         app.py, auth.py, db.py ...
    📥 Reading             app.py
    🔍 pylint  app.py      Score 4.00/10  |  2 issue(s)
    🔒 bandit  app.py      HIGH: 1  MEDIUM: 0  LOW: 0
    💬 Inline comment →    app.py  line 24
    📝 Summary →           Posted to GitHub PR #1
  ──────────────────────────────────────────────────────────────────────
    📋  RAW TOOL FINDINGS  (input to AI review)
  ──────────────────────────────────────────────────────────────────────
    FILE        LINE   TYPE      SEVERITY   CODE / MESSAGE
    app.py      24     security  HIGH       Use of hard-coded password
    db.py       12     pylint    style      missing-module-docstring
  ──────────────────────────────────────────────────────────────────────
    📖  LEGEND
  ──────────────────────────────────────────────────────────────────────
    Pylint score    10/10 = perfect  |  7+/10 = good  |  <5/10 = needs work
    Bandit HIGH   — Serious security vulnerability. Fix before merge.
    Bandit MEDIUM — Potential security risk. Review carefully.
    Bandit LOW    — Minor security note. Use your judgement.
    Style         — Code quality, PEP8, or complexity issue.
  ──────────────────────────────────────────────────────────────────────
    ✅  Review complete — check GitHub PR #1 for comments.
  ══════════════════════════════════════════════════════════════════════

GitHub PR shows the AI's actual review:
  - Conversation tab  → 🤖 AI Code Review Summary (Bugs, Security,
                        Style, What looks good, Recommendation)
  - Files changed tab → Inline comments on specific lines of code


AFTER THE DEMO
───────────────
  - Close the PR on GitHub without merging (Close pull request button)
  - The branch can be deleted too (GitHub prompts you after closing)
  - For next demo: create a new branch and repeat from Step 5


TROUBLESHOOTING
────────────────
  403 on webhook ping
    → Server not running. Start python server.py first, then redeliver.

  422 on inline comment
    → Line not in diff. Now handled gracefully with fallback comment.

  Agent runs but no GitHub comments
    → Check GITHUB_TOKEN is set correctly (source setup_env.sh)
    → Check token has Pull requests: Read and Write permission

  json.loads error / empty body
    → GitHub webhook Content type must be application/json (not form-encoded)

  FileNotFoundError for tests/xxx.py
    → Check linter.py and security.py use os.path.basename(filename)

  ngrok 403
    → Free ngrok URL expired. Restart ngrok, update webhook URL on GitHub.

  echo $OPENAI_API_KEY prints nothing
    → You used bash instead of source. Run: source setup_env.sh

  OpenAI 401 error
    → API key invalid or not set. Check setup_env.sh values.


KEY TEACHING POINTS FOR THE AUDIENCE
──────────────────────────────────────
  1. The agent decides which tools to call and in what order — it is
     not scripted. This is the "agentic" behaviour.

  2. The terminal shows RAW TOOL FINDINGS — what pylint and bandit found.
     The GitHub comments show the AI's INTERPRETATION of those findings.
     These are two different things.

  3. context.txt in the repo root lets the agent understand the project
     without hardcoding anything. Change it to adapt to any project.

  4. The agent uses your GitHub token to post comments as if it were
     a human reviewer — it is a participant in the PR workflow.

  5. Extending this: multi-agent (specialist agents per concern),
     auto-fix PRs, memory across reviews, custom team coding standards.


FILES TO KEEP UPDATED
──────────────────────
  setup_env.sh          → if tokens/keys change
  context.txt           → if project structure changes (lives in repo)
  ngrok webhook URL     → every time ngrok is restarted (free tier)
═══════════════════════════════════════════════════════════════════════
