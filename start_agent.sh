#!/bin/bash
set -e

# Resolve this script's own directory so it works regardless of where you run it from
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "🚀 Starting TrustTrade AI Agent on port 8000..."

# Python discovery order:
#  1. Active virtual env (venv / conda activate)
#  2. Conda base env (common on macOS dev machines)
#  3. System python3 (last resort - may lack deps)
if [ -n "$VIRTUAL_ENV" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python3"
elif [ -n "$CONDA_PREFIX" ]; then
    PYTHON="$CONDA_PREFIX/bin/python3"
elif command -v conda &>/dev/null; then
    CONDA_BASE="$(conda info --base 2>/dev/null)"
    PYTHON="${CONDA_BASE}/bin/python3"
else
    PYTHON="$(command -v python3)"
fi

if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
    echo "❌ Could not find a usable python3."
    echo "   Activate your conda or virtual environment first, then re-run."
    exit 1
fi

# Verify uvicorn is available under this python
if ! "$PYTHON" -m uvicorn --version &>/dev/null; then
    echo "❌ uvicorn not found under: $PYTHON"
    echo "   Run: pip install -r requirements.txt"
    exit 1
fi

echo "   Python  : $PYTHON"
echo "   Dir     : $SCRIPT_DIR"
echo ""

# Use Render's PORT or default to 8000
TARGET_PORT="${PORT:-8000}"

exec "$PYTHON" -m uvicorn main:app --host 0.0.0.0 --port "$TARGET_PORT" --reload
