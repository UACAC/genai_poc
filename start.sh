#!/bin/bash
cd "$(dirname "$0")"

# Determine which docker compose command to use
if command -v docker-compose &> /dev/null; then
    docker-compose up --build -d
elif docker compose version &> /dev/null; then
    docker compose up --build -d
else
    echo "Error: Neither docker-compose nor docker compose found"
    exit 1
fi

echo "JITC GENAI Validation System started. Access the UI at http://localhost:8501"
