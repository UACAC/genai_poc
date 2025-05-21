#!/bin/bash
cd "$(dirname "$0")"

# Determine which docker compose command to use
if command -v docker-compose &> /dev/null; then
    docker-compose down -v
elif docker compose version &> /dev/null; then
    docker compose down -v
else
    echo "Error: Neither docker-compose nor docker compose found"
    exit 1
fi

echo "JITC GENAI Validation System stopped."
