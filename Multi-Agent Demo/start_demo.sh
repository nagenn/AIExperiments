#!/bin/bash
# start_demo.sh — opens all 5 terminals in one command
# Usage: ./start_demo.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KEY="${OPENAI_API_KEY:-}"

if [ -z "$KEY" ]; then
  echo ""
  echo "  ❌  OPENAI_API_KEY not set."
  echo "      Run: export OPENAI_API_KEY=your-key-here"
  echo "      Then: ./start_demo.sh"
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
  echo "  No terminal launcher detected. Open 5 terminals manually:"
  echo "    Terminal 1: python3.9 agents/contract_intelligence.py --service"
  echo "    Terminal 2: python3.9 agents/risk_benchmarking.py --service"
  echo "    Terminal 3: python3.9 agents/policy_compliance.py --service"
  echo "    Terminal 4: python3.9 agents/negotiation_advisor.py --service"
  echo "    Terminal 5: python3.9 demo.py"
  exit 0
fi

launch_mac() {
  osascript <<EOF
tell application "Terminal"
  do script "cd '$SCRIPT_DIR' && export OPENAI_API_KEY='$KEY' && $2"
  set custom title of front window to "$1"
end tell
EOF
  sleep 0.8
}

launch_gnome() {
  gnome-terminal --title="$1" -- bash -c \
    "cd '$SCRIPT_DIR' && export OPENAI_API_KEY='$KEY' && $2; exec bash" &
  sleep 0.5
}

launch_xterm() {
  xterm -title "$1" -fa 'Monospace' -fs 11 -e bash -c \
    "cd '$SCRIPT_DIR' && export OPENAI_API_KEY='$KEY' && $2; exec bash" &
  sleep 0.5
}

launch() {
  case $LAUNCHER in
    mac)   launch_mac   "$1" "$2" ;;
    gnome) launch_gnome "$1" "$2" ;;
    xterm) launch_xterm "$1" "$2" ;;
  esac
}

echo "  Starting agent terminals..."
launch "1 — Contract Intelligence" "python3.9 agents/contract_intelligence.py --service"
launch "2 — Risk & Benchmarking"   "python3.9 agents/risk_benchmarking.py --service"
launch "3 — Policy & Compliance"   "python3.9 agents/policy_compliance.py --service"
launch "4 — Negotiation Advisor"   "python3.9 agents/negotiation_advisor.py --service"
sleep 1
echo "  Starting demo controller..."
launch "5 — DEMO CONTROLLER"       "python3.9 demo.py"

echo ""
echo "  ✅  All 5 terminals open."
echo "  Use Terminal 5 to drive the demo."
echo ""
