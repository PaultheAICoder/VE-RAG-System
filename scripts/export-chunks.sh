#!/bin/bash
# Export all RAG document chunks and Ollama-generated questions to Excel.
#
# Output: data/chunks_MM_DD_YYYY.xlsx
#
# Usage:
#   ./scripts/export-chunks.sh
#   ./scripts/export-chunks.sh -o custom_output.xlsx

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
else
    echo "ERROR: Virtual environment not found at $PROJECT_DIR/.venv"
    echo "  Run: python -m venv .venv && pip install -r requirements-wsl.txt"
    exit 1
fi

# Create data directory if needed
mkdir -p "$PROJECT_DIR/data"

# Run the export
cd "$PROJECT_DIR"
python scripts/export_chunks.py "$@"
