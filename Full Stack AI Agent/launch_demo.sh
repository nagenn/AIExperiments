#!/bin/bash
# launch_demo.sh  —  Support Triage Agent demo launcher (Mac)
#
# Usage:
#   chmod +x launch_demo.sh   (first time only)
#   ./launch_demo.sh
#
# This script:
#   1. Checks your OPENAI_API_KEY is set
#   2. Opens Terminal Window 1 — starts the FastAPI customer history API
#   3. Waits 3 seconds for the API to be ready
#   4. Opens Terminal Window 2 — runs the triage agent demo

# ---------------------------------------------------------------------------
# Get the directory where this script lives (works regardless of where you
# call it from)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Check OPENAI_API_KEY is available
# ---------------------------------------------------------------------------
if [ -z "$OPENAI_API_KEY" ]; then
  echo "❌  ERROR: OPENAI_API_KEY is not set in your environment."
  echo "    Run:  export OPENAI_API_KEY=sk-..."
  echo "    Then re-run this script."
  exit 1
fi

echo "✅  OPENAI_API_KEY found."

# ---------------------------------------------------------------------------
# Check support_history.db exists — remind user to seed if not
# ---------------------------------------------------------------------------
if [ ! -f "$SCRIPT_DIR/support_history.db" ]; then
  echo ""
  echo "⚠️   support_history.db not found."
  echo "    You need to seed the database first. Run:"
  echo "    cd \"$SCRIPT_DIR\" && python3.9 seed_db.py"
  echo ""
  read -p "    Run seed_db.py now? (y/n): " SEED
  if [ "$SEED" = "y" ] || [ "$SEED" = "Y" ]; then
    echo "    Seeding database — this takes about 90 seconds..."
    cd "$SCRIPT_DIR" && python3.9 seed_db.py
    if [ $? -ne 0 ]; then
      echo "❌  seed_db.py failed. Check your OPENAI_API_KEY and try again."
      exit 1
    fi
  else
    echo "    Exiting. Please run seed_db.py before launching the demo."
    exit 1
  fi
fi

echo "✅  Database found."

# ---------------------------------------------------------------------------
# Window 1 — FastAPI customer history API
# ---------------------------------------------------------------------------
echo ""
echo "🚀  Opening Terminal 1 — Customer History API (port 8000)..."

osascript <<EOF
tell application "Terminal"
  activate
  set apiWindow to do script "export PATH='$PATH'; export OPENAI_API_KEY='$OPENAI_API_KEY'; cd '$SCRIPT_DIR'; echo ''; echo '========================================'; echo '  Customer History API — Terminal 1'; echo '========================================'; uvicorn api:app --reload --port 8000"
  set custom title of front window to "Support API — Terminal 1"
end tell
EOF

# ---------------------------------------------------------------------------
# Wait for API to be ready
# ---------------------------------------------------------------------------
echo "⏳  Waiting for API to start..."
sleep 4

# Quick health check
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null)
if [ "$HTTP_STATUS" = "200" ]; then
  echo "✅  API is up and healthy."
else
  echo "⚠️   API health check returned status: $HTTP_STATUS"
  echo "    The API may still be starting — the agent will retry automatically."
fi

# ---------------------------------------------------------------------------
# Window 2 — Triage Agent
# ---------------------------------------------------------------------------
echo ""
echo "🚀  Opening Terminal 2 — Support Triage Agent..."

osascript <<EOF
tell application "Terminal"
  activate
  set agentWindow to do script "export PATH='$PATH'; export OPENAI_API_KEY='$OPENAI_API_KEY'; cd '$SCRIPT_DIR'; echo ''; echo '========================================'; echo '  Support Triage Agent — Terminal 2'; echo '========================================'; echo ''; python3.9 agent.py"
  set custom title of front window to "Triage Agent — Terminal 2"
end tell
EOF

echo ""
echo "✅  Both terminals launched."
echo ""
echo "    Terminal 1 — API server (leave running)"
echo "    Terminal 2 — Agent demo (press Enter between tickets)"
echo ""
echo "    To stop: Ctrl+C in each terminal window when done."
