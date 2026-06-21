#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$ROOT_DIR/.venv" ]; then
  python3 -m venv "$ROOT_DIR/.venv"
fi

"$ROOT_DIR/.venv/bin/python" -m pip install -r "$ROOT_DIR/backend/requirements.txt"

cd "$ROOT_DIR/backend"
echo "Open http://localhost:8000/ui/"
"$ROOT_DIR/.venv/bin/python" -m uvicorn main:app --host 0.0.0.0 --port 8000
