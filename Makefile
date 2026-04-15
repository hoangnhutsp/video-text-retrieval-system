# Video Text Retrieval System - Build & Management Commands

.PHONY: build build-all build-api build-ui build-nginx up down restart clean logs help

# Build all services with no cache
build-all:
	docker compose build --no-cache

# Build specific services with no cache
build-api:
	docker compose build --no-cache video-retrieval-api

build-ui:
	docker compose build --no-cache streamlit-ui

# Build both custom services (API + UI) with no cache  
build:
	docker compose build --no-cache video-retrieval-api streamlit-ui

# Start all services
up:
	docker compose up -d

# Stop all services
down:
	docker compose down

# Restart all services
restart:
	docker compose restart

# Restart specific services
restart-api:
	docker compose restart video-retrieval-api

restart-ui:
	docker compose restart streamlit-ui

restart-nginx:
	docker compose restart nginx

# Full rebuild and restart
rebuild: build-all restart

# Clean up containers, images, and volumes
clean:
	docker compose down -v --rmi all --remove-orphans

# View logs
logs:
	docker compose logs -f

logs-api:
	docker compose logs -f video-retrieval-api

logs-ui:
	docker compose logs -f streamlit-ui

# Check service status
status:
	docker compose ps

# Help command
help:
	@echo "Video Text Retrieval System - Available Commands:"
	@echo ""
	@echo "Build Commands:"
	@echo "  build-all     - Build all services with --no-cache"
	@echo "  build         - Build API and UI services with --no-cache"
	@echo "  build-api     - Build only API service with --no-cache"
	@echo "  build-ui      - Build only UI service with --no-cache"
	@echo ""
	@echo "Service Management:"
	@echo "  up            - Start all services"
	@echo "  down          - Stop all services"
	@echo "  restart       - Restart all services"
	@echo "  restart-api   - Restart API service only"
	@echo "  restart-ui    - Restart UI service only"
	@echo "  restart-nginx - Restart nginx service only"
	@echo "  rebuild       - Full rebuild and restart"
	@echo ""
	@echo "Debugging:"
	@echo "  logs          - View all service logs"
	@echo "  logs-api      - View API service logs"
	@echo "  logs-ui       - View UI service logs"
	@echo "  status        - Check service status"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean         - Remove all containers, images, and volumes"
