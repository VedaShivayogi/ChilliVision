#!/usr/bin/env bash
# ─────────────────────────────────────────
# ChilliVision — Start Server
# ─────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "🌶  ChilliVision — Explainable Chilli Quality Grading"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌  Python 3 not found. Please install Python 3.9+"
    exit 1
fi

echo "📦  Checking dependencies…"
pip install -r requirements.txt -q --break-system-packages 2>/dev/null || \
pip install -r requirements.txt -q

echo "✅  Dependencies OK"
echo ""

# Train model if weights don't exist
if [ ! -f "backend/chilli_net.pth" ]; then
    echo "🧠  No trained model found. Running quick training…"
    python3 train_model.py --epochs 15
    echo ""
fi

echo "🚀  Starting Flask server…"
echo "   Web UI → http://localhost:5000"
echo "   API    → http://localhost:5000/api/analyse"
echo ""

cd backend
python3 app.py
