#!/bin/bash
cd "$(dirname "$0")"
docker-compose up --build -d
echo "✅ GENAI system started: http://localhost:8501"
