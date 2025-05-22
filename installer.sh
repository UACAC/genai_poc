#!/bin/bash
# Platform-agnostic GENAI Validation System setup

# Configuration
REPO_URL="https://github.com/martinmanuel9/dis_verification_genai"

# Get current directory
CURRENT_DIR="$(pwd)"
PROJECT_NAME="$(basename "$CURRENT_DIR")"
echo "Setting up project: $PROJECT_NAME"
echo "Current directory: $CURRENT_DIR"

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

# Confirm model download
echo ""
echo "Would you like to download Llama3, Mistral, and Gemma models now? (y/n)"
read download_answer
if [[ "$download_answer" =~ ^[Yy]$ ]]; then
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

    for model in llama3 mistral gemma; do
        echo "⬇️ Downloading $model..."
        docker run --rm -v "$MODEL_DIR:/root/.ollama" ollama-downloader -c '
        ollama serve &
        for i in {1..10}; do
          if curl -s http://localhost:11434/version; then break; fi
          echo "⏳ Waiting for Ollama to start..."; sleep 3
        done &&
        ollama pull '"$model"'
        ' | tee "$LOGS_DIR/${model}_download.log"

        if [ -d "$MODEL_DIR/models" ] && find "$MODEL_DIR/models" -name "*$model*" | grep -q .; then
            echo "✅ $model downloaded successfully."
        else
            echo "⚠️ $model download failed. Trying fallback method..."
            docker run --rm -v "$MODEL_DIR:/root/.ollama" ollama/ollama pull $model
        fi
    done

    docker rmi ollama-downloader
    rm -rf "$TEMP_DIR"
else
    echo "⏭ Skipping model download."
fi


# Generate Dockerfile.ollama if missing
OLLAMA_DOCKERFILE="$CURRENT_DIR/Dockerfile.ollama"
if [ ! -f "$OLLAMA_DOCKERFILE" ]; then
    echo "📝 Creating Dockerfile.ollama..."
    cat > "$OLLAMA_DOCKERFILE" <<EOL
FROM ollama/ollama:latest
EXPOSE 11434
ENTRYPOINT ["/bin/sh", "-c", "ollama serve & wait"]
EOL
fi

# Add start/stop scripts
cat > "$CURRENT_DIR/start.sh" <<EOL
#!/bin/bash
cd "\$(dirname "\$0")"
$DOCKER_COMPOSE_CMD up --build -d
echo "✅ GENAI system started: http://localhost:8501"
EOL

cat > "$CURRENT_DIR/stop.sh" <<EOL
#!/bin/bash
cd "\$(dirname "\$0")"
$DOCKER_COMPOSE_CMD down -v
echo "🛑 GENAI system stopped."
EOL

chmod +x "$CURRENT_DIR/start.sh" "$CURRENT_DIR/stop.sh"

echo ""
echo "✅ Installation complete!"
echo "▶ To start the system: ./start.sh"
echo "⏹ To stop the system:  ./stop.sh"
echo ""

read -p "Start system now? (y/n): " start_now
if [[ "$start_now" =~ ^[Yy]$ ]]; then
    "$CURRENT_DIR/start.sh"
fi
