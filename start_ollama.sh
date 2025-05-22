#!/bin/sh

# Start Ollama server in the background
ollama serve &

# Wait for server to be available
echo "Waiting for Ollama to become available..."
until curl -s http://localhost:11434 > /dev/null; do
  sleep 1
done

# Pull models
ollama pull llama3
ollama pull mistral
ollama pull gemma

# Keep container alive
wait
