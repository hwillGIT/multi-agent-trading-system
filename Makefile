# Makefile for Multi-Agent Trading System

.PHONY: help install install-dev test test-cov lint format clean docker-build docker-up docker-down logs

# Default target
help:
	@echo "Multi-Agent Trading System - Available Commands:"
	@echo ""
	@echo "Setup Commands:"
	@echo "  install      - Install dependencies"
	@echo "  install-dev  - Install dependencies including dev tools"
	@echo "  setup        - Complete setup including environment"
	@echo ""
	@echo "Development Commands:"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  clean        - Clean cache and build files"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build - Build Docker images"
	@echo "  docker-up    - Start all services"
	@echo "  docker-down  - Stop all services"
	@echo "  logs         - View Docker logs"
	@echo ""
	@echo "Run Commands:"
	@echo "  run          - Run the trading system"
	@echo "  jupyter      - Start Jupyter Lab"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

setup:
	@echo "Setting up Multi-Agent Trading System..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file from template. Please edit with your API keys."; \
	fi
	@mkdir -p logs data/raw data/processed data/features models
	$(MAKE) install-dev
	@echo "Setup complete! Edit .env file and run 'make run' to start."

# Testing
test:
	pytest

test-cov:
	pytest --cov=trading_system --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 .
	mypy .

format:
	black .
	isort .

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/

# Docker commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-restart:
	$(MAKE) docker-down
	$(MAKE) docker-up

logs:
	docker-compose logs -f

# Run commands
run:
	python main.py

jupyter:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Development helpers
create-migration:
	alembic revision --autogenerate -m "$(MSG)"

migrate:
	alembic upgrade head

# Monitoring
monitor:
	@echo "Opening monitoring dashboards..."
	@echo "Grafana: http://localhost:3000 (admin/admin123)"
	@echo "Prometheus: http://localhost:9090"
	@echo "Jupyter: http://localhost:8888"

# Database
db-shell:
	docker-compose exec postgres psql -U postgres -d trading_db

redis-shell:
	docker-compose exec redis redis-cli

# Backup
backup:
	@echo "Creating backup..."
	@mkdir -p backups
	docker-compose exec postgres pg_dump -U postgres trading_db > backups/trading_db_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "Backup created in backups/ directory"

# Security
security-check:
	safety check
	bandit -r .

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "API documentation available in README.md"
	@echo "Configuration guide in config/config.yaml"

# Quick development setup
dev-setup: setup
	@echo "Starting development environment..."
	$(MAKE) docker-up
	@echo "Development environment ready!"
	@echo "Services:"
	@echo "  - Trading System: docker-compose logs -f trading-system"
	@echo "  - Jupyter Lab: http://localhost:8888"
	@echo "  - Grafana: http://localhost:3000"
	@echo "  - Database: make db-shell"

# Production deployment
prod-deploy:
	@echo "Deploying to production..."
	@if [ "$(ENV)" != "production" ]; then \
		echo "Error: Set ENV=production to deploy"; \
		exit 1; \
	fi
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Health check
health-check:
	@echo "Checking system health..."
	@curl -s http://localhost:8000/health || echo "Trading system not responding"
	@docker-compose ps

# Performance test
perf-test:
	@echo "Running performance tests..."
	python -m pytest tests/performance/ -v

# Load sample data
load-sample-data:
	@echo "Loading sample data..."
	python scripts/load_sample_data.py

# Reset environment
reset:
	$(MAKE) docker-down
	docker system prune -f
	rm -rf data/processed/*
	rm -rf logs/*
	$(MAKE) docker-up