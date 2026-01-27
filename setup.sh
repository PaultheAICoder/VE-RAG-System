#!/bin/bash
# Setup script for Agentic RAG on DGX Spark

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Agentic RAG Setup for DGX Spark ==="
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Verify Ollama is accessible
echo ""
echo "Checking Ollama connection..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Ollama is running"

    # Check for required models
    echo ""
    echo "Checking for required models..."

    if ollama list | grep -q "nomic-embed-text"; then
        echo "  nomic-embed-text: installed"
    else
        echo "  nomic-embed-text: NOT FOUND - Installing..."
        ollama pull nomic-embed-text
    fi

    if ollama list | grep -q "llama3.2"; then
        echo "  llama3.2: installed"
    else
        echo "  llama3.2: NOT FOUND - Installing..."
        ollama pull llama3.2:latest
    fi
else
    echo "WARNING: Ollama is not running at localhost:11434"
    echo "Please ensure Ollama is running before starting the app"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the Agentic RAG server:"
echo "  cd $SCRIPT_DIR"
echo "  source .venv/bin/activate"
echo "  python app.py"
echo ""
echo "Or use: ./start.sh"
echo ""
echo "Access the web UI at: http://localhost:8501"
