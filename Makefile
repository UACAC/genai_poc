.PHONY: build run compose-up compose-down clean all remove

# List all Docker Compose YAML files in the current directory
COMPOSE_FILES := $(wildcard *.yml)

# Build the Docker image
build:
	docker-compose up --build || docker compose up --build

# Start Docker Compose services (Milvus and PostgreSQL)
compose-up:
	docker-compose up || docker compose up

# Stop Docker Compose services
# compose-down:
# 	docker-compose down --rmi all

# Clean up Docker images
clean:
	docker rmi jitc-genai-validation

# Build and run in one command
all: build status 

# Remove Docker container and image
remove:
	docker ps -q -f ancestor=jitc-genai-validation | xargs -r docker stop
	docker ps -a -q -f ancestor=jitc-genai-validation | xargs -r docker rm
	docker rmi jitc-genai-validation

# Stop and remove all containers and images
clean-all: compose-down --rmi all || docker compose down --rmi all


status:
	@echo "-----------------------------------------------------------------------------------------------------------------------------------------"
	@docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Image}}\t{{.Ports}}"; 
	@echo "-----------------------------------------------------------------------------------------------------------------------------------------"\
	done
