#!/bin/bash
cd "$(dirname "$0")"
docker-compose down -v
echo "🛑 GENAI system stopped."
