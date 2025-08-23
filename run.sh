#!/bin/bash

# Check for command line arguments
if [ "$1" = "check" ] || [ "$1" = "quality" ]; then
    echo "Running quality checks..."
    ./dev.sh check
    exit 0
fi

if [ "$1" = "quick-check" ]; then
    echo "Running quick quality checks (without tests)..."
    ./dev.sh quick
    exit 0
fi

# Create necessary directories
mkdir -p docs 

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "Error: backend directory not found"
    exit 1
fi

echo "Starting Course Materials RAG System..."
echo "Make sure you have set your ANTHROPIC_API_KEY in .env"
echo ""
echo "Tip: Use './run.sh check' to run quality checks"
echo "      Use './run.sh quick-check' to run checks without tests"

# Change to backend directory and start the server
cd backend && uv run uvicorn app:app --reload --port 8000