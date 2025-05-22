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
	docker rmi dis_verification_genai

# Build and run in one command
all: build status 

# Remove Docker container and image
remove:
	docker ps -q -f ancestor=dis_verification_genai| xargs -r docker stop
	docker ps -a -q -f ancestor=dis_verification_genai | xargs -r docker rm
	docker rmi dis_verification_genai

# Stop and remove all containers and images
clean-all: docker compose-down all || docker compose down all


status:
	@echo "-----------------------------------------------------------------------------------------------------------------------------------------"
	@docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Image}}\t{{.Ports}}"; 
	@echo "-----------------------------------------------------------------------------------------------------------------------------------------"\
	done
