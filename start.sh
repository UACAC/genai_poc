#!/bin/bash
# AI system startup script

cd "$(dirname "$0")"

echo "Starting AI System..."
echo "Location: $(pwd)"

# Check if .env exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found"
fi

# Start services
echo "Starting Docker services..."
docker build base-poetry-deps
docker compose up --build -d

echo ""
echo "Waiting for services to initialize..."
sleep 10

# Check service health
echo "Checking service status..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Ollama service: Running"
else
    echo "Ollama service: Starting (may take a few minutes)"
fi

if curl -s http://localhost:9020/health > /dev/null; then
    echo "FastAPI service: Running"
else
    echo "FastAPI service: Starting"
fi

if curl -s http://localhost:8501 > /dev/null; then
    echo "Streamlit service: Running"
else
    echo "Streamlit service: Starting"
fi

echo ""
echo "AI system startup complete!"
echo "Access points:"
echo "   Streamlit UI: http://localhost:8501"
echo "   FastAPI: http://localhost:9020"
echo "   Ollama API: http://localhost:11434"
echo "   ChromaDB: http://localhost:8020"
echo ""
echo "To check logs: docker-compose logs -f [service_name]"
echo "To stop system: ./stop.sh"
