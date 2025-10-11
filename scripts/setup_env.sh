#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: Python interpreter '$PYTHON_BIN' not found. Override with PYTHON_BIN=/path/to/python." >&2
  exit 1
fi

echo "Using virtual environment directory: $VENV_DIR"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "Reusing existing virtual environment."
fi

VENV_PYTHON="$VENV_DIR/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Error: Python executable not found in $VENV_DIR/bin." >&2
  exit 1
fi

echo "Upgrading pip..."
"$VENV_PYTHON" -m pip install --upgrade pip

if [[ -f "$ROOT_DIR/requirements.txt" ]]; then
  echo "Installing requirements..."
  "$VENV_PYTHON" -m pip install -r "$ROOT_DIR/requirements.txt"
fi

if [[ -f "$ROOT_DIR/setup.py" || -f "$ROOT_DIR/pyproject.toml" ]]; then
  echo "Installing project in editable mode..."
  "$VENV_PYTHON" -m pip install -e "$ROOT_DIR"
fi

cat <<EOF

Virtual environment ready at $VENV_DIR
Activate it with:
  source "$VENV_DIR/bin/activate"

EOF
