#!/bin/bash
# Model management utility

cd "$(dirname "$0")"

case "$1" in
    "list")
        echo "Available models in Ollama:"
        docker exec ollama ollama list
        ;;
    "pull")
        if [ -z "$2" ]; then
            echo "Usage: ./models.sh pull <model_name>"
            exit 1
        fi
        echo "Pulling model: $2"
        docker exec ollama ollama pull "$2"
        ;;
    "remove")
        if [ -z "$2" ]; then
            echo "Usage: ./models.sh remove <model_name>"
            exit 1
        fi
        echo "Removing model: $2"
        docker exec ollama ollama rm "$2"
        ;;
    *)
        echo "Model Management Utility"
        echo "Usage:"
        echo "  ./models.sh list           - List all models"
        echo "  ./models.sh pull <model>   - Pull a specific model"
        echo "  ./models.sh remove <model> - Remove a specific model"
        ;;
esac
