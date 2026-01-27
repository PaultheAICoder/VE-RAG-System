#!/bin/bash
# Start the Agentic RAG server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source .venv/bin/activate

# Set environment variables (can be overridden)
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"
export CHAT_MODEL="${CHAT_MODEL:-llama3.2:latest}"
export CHROMA_PERSIST_DIR="${CHROMA_PERSIST_DIR:-./chroma_db}"

echo "Starting Agentic RAG server..."
echo "  Ollama URL: $OLLAMA_BASE_URL"
echo "  Chat Model: $CHAT_MODEL"
echo "  Embedding Model: $EMBEDDING_MODEL"
echo "  Web UI: http://0.0.0.0:8501"
echo ""

python app.py
