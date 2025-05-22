#!/bin/bash
# JITC GENAI Validation System - Validation Script
# This script verifies the installation and runs endpoint tests

# Configuration
PROJECT_NAME="$(basename "$PWD")"
PROJECT_DIR="$HOME/$PROJECT_NAME"
MODEL_DIR="$PROJECT_DIR/models"
CHROMADB_DIR="$PROJECT_DIR/chromadb"
LOGFILE="$PROJECT_DIR/logs/validation_$(date +%Y%m%d_%H%M%S).log"

# Make sure log directory exists
mkdir -p "$(dirname "$LOGFILE")"

# Helper functions
log() {
    echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOGFILE"
}

check_status() {
    if [ $1 -eq 0 ]; then
        log "✅ $2"
        return 0
    else
        log "❌ $2 (Failed)"
        return 1
    fi
}

print_section() {
    log "==============================================" 
    log "📋 $1" 
    log "=============================================="
}

# Check if script is running from the project directory
if [ ! -f "./start.sh" ] || [ ! -f "./stop.sh" ]; then
    echo "Error: Please run this script from the project directory containing start.sh and stop.sh"
    exit 1
fi

print_section "JITC GENAI Validation System - Validation"
log "Starting validation at $(date)"
log "Project directory: $PROJECT_DIR"

# STEP 1: Check if required files exist
print_section "Checking Required Files"

# Check Docker files
files_to_check=(
    "docker-compose.yml"
    "Dockerfile.ollama"
    ".env"
    "start.sh"
    "stop.sh"
)

missing_files=0
for file in "${files_to_check[@]}"; do
    if [ -f "$file" ]; then
        log "✅ Found $file"
    else
        log "❌ Missing $file"
        missing_files=$((missing_files+1))
    fi
done

if [ $missing_files -gt 0 ]; then
    log "Error: $missing_files required files are missing. Please reinstall."
    exit 1
fi

# STEP 2: Check if models are downloaded
print_section "Checking Downloaded Models"

# Make sure model directory exists
if [ ! -d "$MODEL_DIR/models" ]; then
    log "❌ Models directory not found at $MODEL_DIR/models"
    log "Would you like to run the model download process? (y/n)"
    read -r download_answer
    if [ "$download_answer" == "y" ] || [ "$download_answer" == "Y" ]; then
        log "Please run the installer script again and select 'y' when asked to download models."
        exit 1
    else
        log "Skipping model validation."
    fi
else
    # Check for each model
    models_missing=0
    for model in llama3 mistral gemma; do
        if find "$MODEL_DIR/models" -name "*$model*" | grep -q .; then
            log "✅ $model model files found"
        else
            log "❌ $model model files NOT found"
            models_missing=$((models_missing+1))
        fi
    done
    
    if [ $models_missing -gt 0 ]; then
        log "Warning: $models_missing models appear to be missing."
        log "Would you like to continue validation anyway? (y/n)"
        read -r continue_answer
        if [ "$continue_answer" != "y" ] && [ "$continue_answer" != "Y" ]; then
            log "Validation aborted."
            exit 1
        fi
    else
        log "All required models found."
    fi
fi

# STEP 3: Check if Docker containers are running
print_section "Checking Docker Services"

docker ps >/dev/null 2>&1
if [ $? -ne 0 ]; then
    log "❌ Docker is not running or not accessible"
    log "Please start Docker Desktop or Docker service"
    exit 1
fi

# Check if our containers are running
log "Current running containers:"
docker ps | grep -E 'llama|chromadb|postgres|streamlit' | tee -a "$LOGFILE"

# Count our containers
container_count=$(docker ps | grep -E 'llama|chromadb|postgres|streamlit' | wc -l)

if [ "$container_count" -lt 3 ]; then
    log "❌ Not all required containers are running (expected at least 3)"
    log "Would you like to start the system now? (y/n)"
    read -r start_answer
    if [ "$start_answer" == "y" ] || [ "$start_answer" == "Y" ]; then
        log "Starting system using ./start.sh"
        ./start.sh
        sleep 10  # Give containers time to start
    else
        log "Skipping system start."
        log "Note: ChromaDB endpoint tests will fail if the system is not running."
    fi
else
    log "✅ Required containers appear to be running"
fi

# STEP 4: Run ChromaDB endpoint tests
print_section "Running ChromaDB Endpoint Tests"

if [ ! -f "$CHROMADB_DIR/test_endpoints.py" ]; then
    log "❌ ChromaDB test script not found at: $CHROMADB_DIR/test_endpoints.py"
    
    # Check if it's in the current directory instead
    if [ -f "./chromadb/test_endpoints.py" ]; then
        CHROMADB_DIR="./chromadb"
        log "✅ Found test script at: $CHROMADB_DIR/test_endpoints.py"
    else
        log "Searching for test_endpoints.py..."
        TEST_SCRIPT_PATH=$(find . -name "test_endpoints.py" -type f | head -1)
        
        if [ -n "$TEST_SCRIPT_PATH" ]; then
            CHROMADB_DIR=$(dirname "$TEST_SCRIPT_PATH")
            log "✅ Found test script at: $TEST_SCRIPT_PATH"
        else
            log "❌ Could not find test_endpoints.py anywhere in the project"
            log "Please ensure the test script is available"
            
            # Create a copy of the test script if we have it
            if [ -n "$TEST_SCRIPT_PATH" ] || [ -f "/paste-2.txt" ]; then
                mkdir -p "./chromadb"
                cp "paste-2.txt" "./chromadb/test_endpoints.py" 2>/dev/null || echo "content of paste-2.txt" > "./chromadb/test_endpoints.py"
                CHROMADB_DIR="./chromadb"
                log "Created test script at: $CHROMADB_DIR/test_endpoints.py"
            else
                log "Skipping ChromaDB tests"
                exit 1
            fi
        fi
    fi
fi

# Run the tests
log "Running ChromaDB endpoint tests from $CHROMADB_DIR/test_endpoints.py"
log "This may take a few moments..."

cd "$CHROMADB_DIR" || cd "$(dirname "$TEST_SCRIPT_PATH")" || { log "Could not change to test script directory"; exit 1; }

# Check if Python is available
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    log "❌ Python not found. Cannot run tests."
    log "Please install Python to run the ChromaDB tests"
    exit 1
fi

# Run the tests with either python3 or python
if command -v python3 &>/dev/null; then
    python3 test_endpoints.py > "$LOGFILE.endpoints" 2>&1
    TEST_EXIT_CODE=$?
else
    python test_endpoints.py > "$LOGFILE.endpoints" 2>&1
    TEST_EXIT_CODE=$?
fi

# Display test results
cat "$LOGFILE.endpoints" | tee -a "$LOGFILE"

if [ $TEST_EXIT_CODE -eq 0 ]; then
    log "✅ ChromaDB endpoint tests PASSED"
else
    log "❌ ChromaDB endpoint tests FAILED (exit code: $TEST_EXIT_CODE)"
    log "Please check the logs for more details"
fi

# STEP 5: Summarize validation results
print_section "Validation Summary"

log "Validation completed at $(date)"
log "Log file: $LOGFILE"

if [ $missing_files -eq 0 ] && [ $models_missing -eq 0 ] && [ $container_count -ge 3 ] && [ $TEST_EXIT_CODE -eq 0 ]; then
    log "✅ All validation checks PASSED"
    log "GENAI Validation System appears to be correctly installed and functional."
else
    log "⚠️ Some validation checks FAILED"
    
    if [ $missing_files -gt 0 ]; then
        log "  - Missing required files: $missing_files"
    fi
    
    if [ $models_missing -gt 0 ]; then
        log "  - Missing AI models: $models_missing"
    fi
    
    if [ $container_count -lt 3 ]; then
        log "  - Not all containers are running"
    fi
    
    if [ $TEST_EXIT_CODE -ne 0 ]; then
        log "  - ChromaDB endpoint tests failed"
    fi
    
    log "Please address the issues above to ensure system functionality."
fi

log "To start the system: ./start.sh"
log "To stop the system: ./stop.sh"
log "To rerun validation: ./validation.sh"

# Return to original directory
cd - >/dev/null