#!/bin/bash
# Stop VE-RAG-System servers
# Usage: ./scripts/stop-servers.sh

echo "Stopping VE-RAG-System servers..."

# Kill backend (port 8502)
if lsof -ti:8502 > /dev/null 2>&1; then
    lsof -ti:8502 | xargs kill -9
    echo "Backend (port 8502) stopped"
else
    echo "Backend (port 8502) not running"
fi

# Kill frontend dev server (port 5173)
if lsof -ti:5173 > /dev/null 2>&1; then
    lsof -ti:5173 | xargs kill -9
    echo "Frontend (port 5173) stopped"
else
    echo "Frontend (port 5173) not running"
fi

echo "Done"
