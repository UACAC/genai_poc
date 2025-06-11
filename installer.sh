#!/bin/bash
# Platform-agnostic Legal GENAI Validation System setup

# Configuration
REPO_URL="https://github.com/martinmanuel9/litigation_genai.git"

# Get current directory
CURRENT_DIR="$(pwd)"
PROJECT_NAME="$(basename "$CURRENT_DIR")"
echo "🏛️ Setting up Legal AI project: $PROJECT_NAME"
echo "📁 Current directory: $CURRENT_DIR"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found!"
    echo "Please install Docker Desktop: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check Docker Compose
DOCKER_COMPOSE_CMD=""
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker compose"
else
    echo "❌ Neither docker-compose nor docker compose found!"
    exit 1
fi
echo "✅ Using compose command: $DOCKER_COMPOSE_CMD"

# Check for .env
if [ ! -f "$CURRENT_DIR/.env" ]; then
    echo "❌ .env file not found in $CURRENT_DIR"
    echo "Please create it before continuing."
    exit 1
fi
echo "✅ Found .env file"

# Skip repo clone if local source exists
if [ -f "$CURRENT_DIR/docker-compose.yml" ]; then
    echo "✅ Project already present. Skipping git clone."
else
    echo "📦 Cloning repository from $REPO_URL..."
    git clone "$REPO_URL" "$CURRENT_DIR"
    if [ $? -ne 0 ]; then
        echo "❌ Failed to clone. Please check your Git setup or clone manually."
        exit 1
    fi
fi

# Create data/model/logs directories
echo "📁 Creating necessary directories..."
mkdir -p "$CURRENT_DIR/data/chromadb"
mkdir -p "$CURRENT_DIR/data/postgres"
mkdir -p "$CURRENT_DIR/data/huggingface_cache"
mkdir -p "$CURRENT_DIR/models"
mkdir -p "$CURRENT_DIR/logs"

# Define model categories
# GENERAL_MODELS=("llama3" "mistral" "gemma")
# LEGAL_MODELS=("llama3:8b" "mixtral:8x7b" "codellama:7b")
GENERAL_MODELS=("llama3")

# Function to download models
download_models() {
    local models=("$@")
    local category=$1
    shift
    
    echo "📚 Downloading $category models..."
    
    TEMP_DIR=$(mktemp -d)
    LOGS_DIR="$CURRENT_DIR/logs"
    MODEL_DIR="$CURRENT_DIR/models"

    echo "🛠 Creating temporary Dockerfile for model download..."
    cat > "$TEMP_DIR/Dockerfile" <<EOL
FROM ollama/ollama:latest
RUN apt-get update && apt-get install -y curl
WORKDIR /app
ENTRYPOINT ["/bin/sh"]
EOL

    docker build -t ollama-downloader "$TEMP_DIR"

    for model in "${models[@]}"; do
        echo "⬇️ Downloading $model..."
        
        # Create safe filename for logs
        safe_model_name=$(echo "$model" | sed 's/:/_/g')
        
        docker run --rm -v "$MODEL_DIR:/root/.ollama" ollama-downloader -c '
        ollama serve &
        for i in {1..15}; do
          if curl -s http://localhost:11434/version; then break; fi
          echo "⏳ Waiting for Ollama to start..."; sleep 3
        done &&
        echo "🔄 Pulling '"$model"'..." &&
        ollama pull '"$model"' &&
        echo "✅ Successfully pulled '"$model"'"
        ' 2>&1 | tee "$LOGS_DIR/${safe_model_name}_download.log"

        # Check if download was successful
        if grep -q "Successfully pulled" "$LOGS_DIR/${safe_model_name}_download.log"; then
            echo "✅ $model downloaded successfully."
        else
            echo "⚠️ $model download may have failed. Check logs for details."
            echo "📋 Fallback: You can manually download with: docker exec ollama_container ollama pull $model"
        fi
    done

    docker rmi ollama-downloader 2>/dev/null || true
    rm -rf "$TEMP_DIR"
}

# Enhanced model download options
echo ""
echo "🤖 Model Download Options:"
echo "1) Download all models"
# echo "2) Download general models only (llama3, mistral, gemma)"
# echo "3) Download legal models only (llama3:8b, mixtral:8x7b, codellama:7b)"
echo "2) Custom selection"
echo "3) Skip model download"
echo ""
read -p "Choose option (1-3): " download_option

case $download_option in
    1)
        echo "📦 Downloading all models..."
        download_models "General" "${GENERAL_MODELS[@]}"
        # download_models "Legal" "${LEGAL_MODELS[@]}"
        ;;
    # 2)
    #     download_models "General" "${GENERAL_MODELS[@]}"
    #     ;;
    # 3)
    #     # download_models "Legal" "${LEGAL_MODELS[@]}"
    #     ;;
    2)
        echo "📋 Available models:"
        echo "General: ${GENERAL_MODELS[*]}"
        # echo "Legal: ${LEGAL_MODELS[*]}"
        echo ""
        read -p "Enter models to download (space-separated): " custom_models
        if [ ! -z "$custom_models" ]; then
            IFS=' ' read -ra SELECTED_MODELS <<< "$custom_models"
            download_models "Custom" "${SELECTED_MODELS[@]}"
        fi
        ;;
    3)
        echo "⏭ Skipping model download."
        ;;
    *)
        echo "❌ Invalid option. Skipping model download."
        ;;
esac

# Generate enhanced Dockerfile.ollama
OLLAMA_DOCKERFILE="$CURRENT_DIR/Dockerfile.ollama"
echo "📝 Creating enhanced Dockerfile.ollama..."
cat > "$OLLAMA_DOCKERFILE" <<EOL
FROM ollama/ollama:latest

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy startup script
COPY start_ollama.sh /start.sh
RUN chmod +x /start.sh

# Expose the API port
EXPOSE 11434

# Start script
ENTRYPOINT ["/bin/sh", "/start.sh"]
EOL

# Generate enhanced start_ollama.sh
START_OLLAMA_SCRIPT="$CURRENT_DIR/start_ollama.sh"
echo "📝 Creating enhanced start_ollama.sh..."
cat > "$START_OLLAMA_SCRIPT" <<EOL
#!/bin/sh
# Enhanced Ollama startup script with legal models

# Start Ollama server in the background
echo "🚀 Starting Ollama server..."
ollama serve &

# Wait for server to be available
echo "⏳ Waiting for Ollama to become available..."
until curl -s http://localhost:11434 > /dev/null; do
  sleep 1
done

echo "✅ Ollama server is ready!"

# Function to pull model with error handling
pull_model() {
    local model=\$1
    echo "📥 Attempting to pull \$model..."
    if ollama pull "\$model" 2>/dev/null; then
        echo "✅ Successfully pulled \$model"
    else
        echo "⚠️ Failed to pull \$model (may not be available)"
    fi
}

# Pull general models
echo "📚 Pulling general models..."
pull_model "llama3"
pull_model "mistral"
pull_model "gemma"


# Pull legal models
echo "⚖️  Pulling legal models..."
pull_model "llama3:8b"
pull_model "mixtral:8x7b"
pull_model "codellama:7b"

echo "🎉 Model pulling complete!"

# List available models
echo "📋 Available models:"
ollama list

# Keep container alive
echo "🔄 Keeping Ollama server running..."
wait
EOL

chmod +x "$START_OLLAMA_SCRIPT"


# Add enhanced start/stop scripts
cat > "$CURRENT_DIR/start.sh" <<EOL
#!/bin/bash
# Enhanced Legal AI system startup script

cd "\$(dirname "\$0")"

echo "🏛️ Starting Legal AI Validation System..."
echo "📍 Location: \$(pwd)"

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️ Warning: .env file not found"
fi

# Start services
echo "🚀 Starting Docker services..."
$DOCKER_COMPOSE_CMD up --build -d

echo ""
echo "⏳ Waiting for services to initialize..."
sleep 10

# Check service health
echo "🔍 Checking service status..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama service: Running"
else
    echo "⚠️ Ollama service: Starting (may take a few minutes)"
fi

if curl -s http://localhost:9020/health > /dev/null; then
    echo "✅ FastAPI service: Running"
else
    echo "⚠️ FastAPI service: Starting"
fi

if curl -s http://localhost:8501 > /dev/null; then
    echo "✅ Streamlit service: Running"
else
    echo "⚠️ Streamlit service: Starting"
fi

echo ""
echo "🎉 Legal AI system startup complete!"
echo "🌐 Access points:"
echo "   • Streamlit UI: http://localhost:8501"
echo "   • FastAPI: http://localhost:9020"
echo "   • Ollama API: http://localhost:11434"
echo "   • ChromaDB: http://localhost:8020"
echo ""
echo "📊 To check logs: docker-compose logs -f [service_name]"
echo "🛑 To stop system: ./stop.sh"
EOL

cat > "$CURRENT_DIR/stop.sh" <<EOL
#!/bin/bash
# Enhanced Legal AI system shutdown script

cd "\$(dirname "\$0")"

echo "🛑 Stopping Legal AI Validation System..."
$DOCKER_COMPOSE_CMD down

echo "🧹 Cleaning up..."
docker system prune -f --volumes 2>/dev/null || true

echo "✅ Legal AI system stopped and cleaned up."
echo "📁 Data preserved in ./data/ directory"
echo "🔄 To restart: ./start.sh"
EOL

# Add utility scripts
cat > "$CURRENT_DIR/models.sh" <<EOL
#!/bin/bash
# Model management utility

cd "\$(dirname "\$0")"

case "\$1" in
    "list")
        echo "📋 Available models in Ollama:"
        docker exec ollama ollama list
        ;;
    "pull")
        if [ -z "\$2" ]; then
            echo "Usage: ./models.sh pull <model_name>"
            exit 1
        fi
        echo "📥 Pulling model: \$2"
        docker exec ollama ollama pull "\$2"
        ;;
    "remove")
        if [ -z "\$2" ]; then
            echo "Usage: ./models.sh remove <model_name>"
            exit 1
        fi
        echo "🗑️ Removing model: \$2"
        docker exec ollama ollama rm "\$2"
        ;;
    *)
        echo "Model Management Utility"
        echo "Usage:"
        echo "  ./models.sh list           - List all models"
        echo "  ./models.sh pull <model>   - Pull a specific model"
        echo "  ./models.sh remove <model> - Remove a specific model"
        ;;
esac
EOL

chmod +x "$CURRENT_DIR/start.sh" "$CURRENT_DIR/stop.sh" "$CURRENT_DIR/models.sh"

echo ""
echo "🎉 Legal AI Installation complete!"
echo ""
echo "📋 Available commands:"
echo "   • ./start.sh    - Start the Legal AI system"
echo "   • ./stop.sh     - Stop the Legal AI system"
echo "   • ./models.sh   - Manage AI models"
echo ""
# echo "📚 System includes:"
# echo "   • General models: llama3, mistral, gemma"
# echo "   • Legal models: llama3:8b, mixtral:8x7b, codellama:7b"
# echo ""

read -p "🚀 Start Legal AI system now? (y/n): " start_now
if [[ "$start_now" =~ ^[Yy]$ ]]; then
    "$CURRENT_DIR/start.sh"
fi