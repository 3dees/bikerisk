#!/usr/bin/env bash
# Auto-activate venv and run Python commands
# Usage: ./run.sh <python-command>
# Example: ./run.sh main.py
# Example: ./run.sh -c "import anthropic; print('OK')"

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$PROJECT_ROOT/venv/bin/python"

# Check if venv exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "ERROR: Virtual environment not found at $PROJECT_ROOT/venv" >&2
    echo "Run: python -m venv venv" >&2
    exit 1
fi

# Load .env if present
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Run Python with venv
exec "$VENV_PYTHON" "$@"
