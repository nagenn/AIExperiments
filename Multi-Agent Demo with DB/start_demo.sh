#!/bin/bash
# start_demo.sh — opens all 6 terminals in one command
# Usage: ./start_demo.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KEY="${OPENAI_API_KEY:-}"
VENV_ACTIVATE="$(dirname "$SCRIPT_DIR")/.venv311/bin/activate"

if [ -z "$KEY" ]; then
  echo ""
  echo "  ❌  OPENAI_API_KEY not set."
  echo "      Run: export OPENAI_API_KEY=your-key-here"
  echo "      Then: ./start_demo.sh"
  echo ""
  exit 1
fi

if [ ! -f "$VENV_ACTIVATE" ]; then
  echo ""
  echo "  ❌  Virtual environment not found at .venv311/"
  echo "      Run: python3.11 -m venv .venv311"
  echo "           source .venv311/bin/activate"
  echo "           pip install openai chromadb httpx PyPDF2 \"mcp[cli]\""
  echo ""
  exit 1
fi

echo ""
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║   CONTRACT REVIEW — MULTI-AGENT AI DEMO  v5         ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo ""

if [[ "$OSTYPE" == "darwin"* ]]; then   LAUNCHER="mac"
elif command -v gnome-terminal &>/dev/null; then LAUNCHER="gnome"
elif command -v xterm &>/dev/null; then  LAUNCHER="xterm"
else
  echo "  No terminal launcher detected. Open 6 terminals manually:"
  echo "    source .venv311/bin/activate"
  echo "    Terminal 6: python chroma_mcp_server.py"
  echo "    Terminal 1: python agents/contract_intelligence.py --service"
  echo "    Terminal 2: python agents/risk_benchmarking.py --service"
  echo "    Terminal 3: python agents/policy_compliance.py --service"
  echo "    Terminal 4: python agents/negotiation_advisor.py --service"
  echo "    Terminal 5: python demo.py"
  exit 0
fi

launch_mac() {
  osascript <<EOF
tell application "Terminal"
  do script "cd '$SCRIPT_DIR' && source '$VENV_ACTIVATE' && export OPENAI_API_KEY='$KEY' && $2"
  set custom title of front window to "$1"
end tell
EOF
  sleep 0.8
}

launch_gnome() {
  gnome-terminal --title="$1" -- bash -c \
    "cd '$SCRIPT_DIR' && source '$VENV_ACTIVATE' && export OPENAI_API_KEY='$KEY' && $2; exec bash" &
  sleep 0.5
}

launch_xterm() {
  xterm -title "$1" -fa 'Monospace' -fs 11 -e bash -c \
    "cd '$SCRIPT_DIR' && source '$VENV_ACTIVATE' && export OPENAI_API_KEY='$KEY' && $2; exec bash" &
  sleep 0.5
}

launch() {
  case $LAUNCHER in
    mac)   launch_mac   "$1" "$2" ;;
    gnome) launch_gnome "$1" "$2" ;;
    xterm) launch_xterm "$1" "$2" ;;
  esac
}

echo "  Starting memory server first..."
launch "6 — Chroma Memory"         "python chroma_mcp_server.py"
sleep 2

echo "  Starting agent terminals..."
launch "1 — Contract Intelligence" "python agents/contract_intelligence.py --service"
launch "2 — Risk & Benchmarking"   "python agents/risk_benchmarking.py --service"
launch "3 — Policy & Compliance"   "python agents/policy_compliance.py --service"
launch "4 — Negotiation Advisor"   "python agents/negotiation_advisor.py --service"
sleep 1
echo "  Starting demo controller..."
launch "5 — DEMO CONTROLLER"       "python demo.py"

echo ""
echo "  ✅  All 6 terminals open."
echo "  Terminal 6 = Chroma Memory MCP server — http://localhost:8765/sse"
echo "  (Terminal 6 must stay running for memory to work.)"
echo "  Use Terminal 5 to drive the demo."
echo ""
