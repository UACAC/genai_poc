#!/bin/bash
# Platform-agnostic GENAI Validation System setup

# Configuration
REPO_URL="https://github.com/UACAC/genai_poc.git"

# Get current directory
CURRENT_DIR="$(pwd)"
PROJECT_NAME="$(basename "$CURRENT_DIR")"
echo "Setting up AI project: $PROJECT_NAME"
echo "Current directory: $CURRENT_DIR"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "Docker not found!"
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
    echo "Neither docker-compose nor docker compose found!"
    exit 1
fi
echo "Using compose command: $DOCKER_COMPOSE_CMD"

# Check for .env
if [ ! -f "$CURRENT_DIR/.env" ]; then
    echo ".env file not found in $CURRENT_DIR"
    echo "Please create it before continuing."
    exit 1
fi
echo "Found .env file"

# Skip repo clone if local source exists
if [ -f "$CURRENT_DIR/docker-compose.yml" ]; then
    echo "Project already present. Skipping git clone."
else
    echo "Cloning repository from $REPO_URL..."
    git clone "$REPO_URL" "$CURRENT_DIR"
    if [ $? -ne 0 ]; then
        echo "❌ Failed to clone. Please check your Git setup or clone manually."
        exit 1
    fi
fi

# Create data/model/logs directories
echo "Creating necessary directories..."
mkdir -p "$CURRENT_DIR/data/chromadb"
mkdir -p "$CURRENT_DIR/data/postgres"
mkdir -p "$CURRENT_DIR/data/huggingface_cache"
mkdir -p "$CURRENT_DIR/models"
mkdir -p "$CURRENT_DIR/logs"

# Create .venv directory for local poetry management
echo "Creating .venv directory for local poetry management..."
if [ ! -d "$CURRENT_DIR/.venv" ]; then
    mkdir -p "$CURRENT_DIR/.venv"
    echo ".venv directory created successfully"
else
    echo ".venv directory already exists"
fi

# Setup Poetry for local development
echo "Setting up Poetry for local development..."

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    
    # Add Poetry to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    echo "Poetry installed. You may need to restart your terminal or run:"
    echo "export PATH=\"\$HOME/.local/bin:\$PATH\""
else
    echo "Poetry is already installed"
fi

# Configure Poetry to use the .venv directory
echo "Configuring Poetry to use local .venv directory..."
poetry config virtualenvs.in-project true
poetry config virtualenvs.path "$CURRENT_DIR/.venv"

# Install dependencies if pyproject.toml exists
if [ -f "$CURRENT_DIR/pyproject.toml" ]; then
    echo "Installing Poetry dependencies..."
    cd "$CURRENT_DIR"
    poetry install
    echo "Poetry dependencies installed successfully"
else
    echo "No pyproject.toml found. Skipping Poetry dependency installation."
fi

GENERAL_MODELS=("llama3")

# Function to download models with better error handling
download_models() {
    local models=("$@")
    local category=$1
    shift
    
    echo "Downloading $category models..."
    
    TEMP_DIR=$(mktemp -d)
    LOGS_DIR="$CURRENT_DIR/logs"
    MODEL_DIR="$CURRENT_DIR/models"

    echo "Creating optimized temporary Dockerfile for model download..."
    cat > "$TEMP_DIR/Dockerfile" <<EOL
FROM ollama/ollama:latest
RUN apt-get update && apt-get install -y curl
ENV OLLAMA_KEEP_ALIVE=-1
ENV OLLAMA_NUM_PARALLEL=1
WORKDIR /app
ENTRYPOINT ["/bin/sh"]
EOL

    docker build -t ollama-downloader "$TEMP_DIR"

    for model in "${models[@]}"; do
        echo "Downloading $model..."
        
        safe_model_name=$(echo "$model" | sed 's/:/_/g')
        
        docker run --rm -v "$MODEL_DIR:/root/.ollama" ollama-downloader -c '
        export OLLAMA_KEEP_ALIVE=-1
        export OLLAMA_NUM_PARALLEL=1
        ollama serve &
        for i in {1..30}; do
          if curl -s http://localhost:11434/version; then break; fi
          echo "Waiting for Ollama to start..."; sleep 5
        done &&
        echo "Pulling '"$model"'..." &&
        timeout 600 ollama pull '"$model"' &&
        echo "Successfully pulled '"$model"'"
        ' 2>&1 | tee "$LOGS_DIR/${safe_model_name}_download.log"

        if grep -q "Successfully pulled" "$LOGS_DIR/${safe_model_name}_download.log"; then
            echo "$model downloaded successfully."
        else
            echo "$model download may have failed. Check logs for details."
        fi
    done

    docker rmi ollama-downloader 2>/dev/null || true
    rm -rf "$TEMP_DIR"
}

# Enhanced model download options
echo ""
echo "Model Download Options:"
echo "1) Download all models"
echo "2) Custom selection"
echo "3) Skip model download"
echo ""
read -p "Choose option (1-3): " download_option

case $download_option in
    1)
        echo "Downloading all models..."
        download_models "General" "${GENERAL_MODELS[@]}"
        ;;

    2)
        echo "Available models:"
        echo "${GENERAL_MODELS[*]}"
        echo ""
        read -p "Enter models to download (space-separated): " custom_models
        if [ ! -z "$custom_models" ]; then
            IFS=' ' read -ra SELECTED_MODELS <<< "$custom_models"
            download_models "Custom" "${SELECTED_MODELS[@]}"
        fi
        ;;
    3)
        echo "Skipping model download."
        ;;
    *)
        echo "Invalid option. Skipping model download."
        ;;
esac

# Generate enhanced Dockerfile.ollama
OLLAMA_DOCKERFILE="$CURRENT_DIR/Dockerfile.ollama"
echo "Creating enhanced Dockerfile.ollama..."
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
echo "Creating enhanced start_ollama.sh..."
cat > "$START_OLLAMA_SCRIPT" <<EOL
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
    local model=\$1
    echo "Attempting to pull \$model..."
    if ollama pull "\$model" 2>/dev/null; then
        echo "Successfully pulled \$model"
    else
        echo "Failed to pull \$model (may not be available)"
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
EOL

chmod +x "$START_OLLAMA_SCRIPT"


# Add enhanced start/stop scripts
cat > "$CURRENT_DIR/start.sh" <<EOL
#!/bin/bash
# AI system startup script

cd "\$(dirname "\$0")"

echo "Starting AI System..."
echo "Location: \$(pwd)"

# Check if .env exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found"
fi

# Start services
echo "Starting Docker services..."
$DOCKER_COMPOSE_CMD build base-poetry-deps
$DOCKER_COMPOSE_CMD up --build -d

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
EOL

cat > "$CURRENT_DIR/stop.sh" <<EOL
#!/bin/bash
# Enhanced AI system shutdown script

cd "\$(dirname "\$0")"

echo "Stopping AI Validation System..."
$DOCKER_COMPOSE_CMD down

echo "Cleaning up..."
docker system prune -a 2>/dev/null || true

echo "AI system stopped and cleaned up."
echo "Data preserved in ./data/ directory"
echo "To restart: ./start.sh"
EOL

# Add utility scripts
cat > "$CURRENT_DIR/models.sh" <<EOL
#!/bin/bash
# Model management utility

cd "\$(dirname "\$0")"

case "\$1" in
    "list")
        echo "Available models in Ollama:"
        docker exec ollama ollama list
        ;;
    "pull")
        if [ -z "\$2" ]; then
            echo "Usage: ./models.sh pull <model_name>"
            exit 1
        fi
        echo "Pulling model: \$2"
        docker exec ollama ollama pull "\$2"
        ;;
    "remove")
        if [ -z "\$2" ]; then
            echo "Usage: ./models.sh remove <model_name>"
            exit 1
        fi
        echo "Removing model: \$2"
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
echo "AI Installation complete!"
echo ""
echo "Available commands:"
echo "   ./start.sh    - Start the AI system"
echo "   ./stop.sh     - Stop the AI system"
echo "   ./models.sh   - Manage AI models"
echo ""


read -p "Start AI system now? (y/n): " start_now
if [[ "$start_now" =~ ^[Yy]$ ]]; then
    "$CURRENT_DIR/start.sh"
fi