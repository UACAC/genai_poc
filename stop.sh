#!/bin/bash
# Enhanced Legal AI system shutdown script

cd "$(dirname "$0")"

echo "Stopping Legal AI Validation System..."
docker compose down

echo "🧹 Cleaning up..."
docker system prune -f --volumes 2>/dev/null || true

echo "Legal AI system stopped and cleaned up."
echo "Data preserved in ./data/ directory"
echo "To restart: ./start.sh"
