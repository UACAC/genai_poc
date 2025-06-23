#!/bin/sh
# Enhanced Ollama startup script with legal models

# Start Ollama server in the background
echo "Starting Ollama server..."
ollama serve &

# Wait for server to be available
echo "Waiting for Ollama to become available..."
until curl -s http://localhost:11434 > /dev/null; do
  sleep 1
done

echo "Ollama server is ready!"

# Function to pull model with error handling
pull_model() {
    local model=$1
    echo "Attempting to pull $model..."
    if ollama pull "$model" 2>/dev/null; then
        echo "Successfully pulled $model"
    else
        echo "Failed to pull $model (may not be available)"
    fi
}

# Pull general models
echo "Pulling general models..."
pull_model "llama3"
pull_model "llava"

echo "Model pulling complete!"

# List available models
echo "Available models:"
ollama list

# Keep container alive
echo "Keeping Ollama server running..."
wait
