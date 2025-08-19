# Trading Intelligence System Makefile

.PHONY: help setup dev prod test clean lint type-check docker-build docker-up docker-down logs

# Default target
help:
	@echo "Trading Intelligence System - Available Commands:"
	@echo ""
	@echo "Setup Commands:"
	@echo "  setup          - Setup development environment"
	@echo "  setup-venv     - Create Python virtual environment"
	@echo "  install-deps   - Install Python dependencies"
	@echo ""
	@echo "Development Commands:"
	@echo "  dev            - Start development environment"
	@echo "  test           - Run test suite"
	@echo "  lint           - Run linting"
	@echo "  type-check     - Run type checking"
	@echo "  format         - Format code"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build   - Build all Docker images"
	@echo "  docker-up      - Start all services"
	@echo "  docker-down    - Stop all services"
	@echo "  docker-logs    - View service logs"
	@echo ""
	@echo "Production Commands:"
	@echo "  prod           - Deploy to production"
	@echo "  prod-logs      - View production logs"
	@echo ""
	@echo "Utility Commands:"
	@echo "  clean          - Clean up temporary files"
	@echo "  backup         - Backup data and configuration"

# Environment setup
setup: setup-venv install-deps setup-env
	@echo "✅ Development environment setup complete"

setup-venv:
	@echo "🐍 Creating Python virtual environment..."
	python3 -m venv venv
	@echo "✅ Virtual environment created. Activate with: source venv/bin/activate"

install-deps:
	@echo "📦 Installing Python dependencies..."
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	@echo "✅ Dependencies installed"

setup-env:
	@echo "⚙️ Setting up environment configuration..."
	@if [ ! -f .env ]; then cp .env.template .env; echo "📝 Created .env file from template"; fi
	@mkdir -p data/{parquet,backups}
	@mkdir -p logs
	@mkdir -p config/{prometheus,grafana}
	@echo "✅ Environment configuration complete"

# Development
dev: setup
	@echo "🚀 Starting development environment..."
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
	@echo "✅ Development environment running:"
	@echo "  - API Gateway: http://localhost:8080"
	@echo "  - Jupyter Analytics: http://localhost:8888"
	@echo "  - Grafana: http://localhost:3000"
	@echo "  - Prometheus: http://localhost:9090"

# Testing
test:
	@echo "🧪 Running test suite..."
	python -m pytest tests/ -v --cov=. --cov-report=html
	@echo "✅ Tests complete. Coverage report: htmlcov/index.html"

test-agents:
	@echo "🤖 Testing individual agents..."
	python -m pytest tests/agents/ -v

test-integration:
	@echo "🔗 Running integration tests..."
	python -m pytest tests/integration/ -v

# Code quality
lint:
	@echo "🔍 Running linting..."
	flake8 agents/ common/ tests/ --max-line-length=100
	pylint agents/ common/ --disable=C0114,C0115,C0116
	@echo "✅ Linting complete"

type-check:
	@echo "📋 Running type checking..."
	mypy agents/ common/ --ignore-missing-imports
	@echo "✅ Type checking complete"

format:
	@echo "🎨 Formatting code..."
	black agents/ common/ tests/ --line-length=100
	isort agents/ common/ tests/ --profile black
	@echo "✅ Code formatting complete"

# Docker commands
docker-build:
	@echo "🐳 Building Docker images..."
	docker-compose build
	@echo "✅ Docker images built"

docker-up:
	@echo "🚀 Starting Docker services..."
	docker-compose up -d
	@echo "✅ Services started"

docker-down:
	@echo "🛑 Stopping Docker services..."
	docker-compose down
	@echo "✅ Services stopped"

docker-logs:
	@echo "📋 Viewing service logs..."
	docker-compose logs -f

docker-restart:
	@echo "🔄 Restarting services..."
	docker-compose restart

# Production
prod:
	@echo "🌟 Deploying to production..."
	@if [ ! -f .env.prod ]; then echo "❌ .env.prod file not found"; exit 1; fi
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
	@echo "✅ Production deployment complete"

prod-logs:
	@echo "📋 Viewing production logs..."
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml logs -f

# Data management
backup:
	@echo "💾 Creating backup..."
	@mkdir -p backups
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	tar -czf backups/trading_intelligence_$$timestamp.tar.gz data/ config/ .env
	@echo "✅ Backup created: backups/trading_intelligence_$$(date +%Y%m%d_%H%M%S).tar.gz"

restore:
	@echo "🔄 Restoring from backup..."
	@read -p "Enter backup file path: " backup_file; \
	if [ -f "$$backup_file" ]; then \
		tar -xzf "$$backup_file"; \
		echo "✅ Restore complete"; \
	else \
		echo "❌ Backup file not found"; \
	fi

# Database management
db-migrate:
	@echo "🗄️ Running database migrations..."
	alembic upgrade head
	@echo "✅ Migrations complete"

db-reset:
	@echo "⚠️  Resetting database..."
	@read -p "Are you sure? This will delete all data [y/N]: " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		docker-compose exec postgres psql -U trading_user -d trading_intelligence -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"; \
		make db-migrate; \
		echo "✅ Database reset complete"; \
	else \
		echo "❌ Database reset cancelled"; \
	fi

# Monitoring
monitor:
	@echo "📊 Opening monitoring dashboard..."
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"

health-check:
	@echo "🏥 Checking system health..."
	@curl -s http://localhost:8080/health || echo "❌ API Gateway not responding"
	@curl -s http://localhost:3000/api/health || echo "❌ Grafana not responding"
	@curl -s http://localhost:9090/-/healthy || echo "❌ Prometheus not responding"
	@echo "✅ Health check complete"

# Utility commands
clean:
	@echo "🧹 Cleaning up temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	@echo "✅ Cleanup complete"

logs:
	@echo "📋 Viewing application logs..."
	tail -f logs/*.log

# Deployment helpers
deploy-aws:
	@echo "☁️ Deploying to AWS ECS..."
	# TODO: Implement AWS deployment
	@echo "❌ AWS deployment not yet implemented"

deploy-gcp:
	@echo "☁️ Deploying to Google Cloud Run..."
	# TODO: Implement GCP deployment
	@echo "❌ GCP deployment not yet implemented"

# Security
security-scan:
	@echo "🔒 Running security scan..."
	safety check
	bandit -r agents/ common/ -f json -o security-report.json
	@echo "✅ Security scan complete"

# Performance testing
load-test:
	@echo "⚡ Running load tests..."
	locust -f tests/load/locustfile.py --host=http://localhost:8080
	@echo "✅ Load test complete"

# Documentation
docs:
	@echo "📚 Generating documentation..."
	sphinx-build -b html docs/ docs/_build/html
	@echo "✅ Documentation generated: docs/_build/html/index.html"

# Quick start for new developers
quickstart: setup docker-build docker-up
	@echo ""
	@echo "🎉 Quick start complete!"
	@echo ""
	@echo "Your trading intelligence system is now running:"
	@echo "  📊 Dashboard: http://localhost:3000 (admin/admin)"
	@echo "  🔌 API: http://localhost:8080"
	@echo "  📓 Analytics: http://localhost:8888"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Configure your data sources in .env"
	@echo "  2. Review the API documentation at http://localhost:8080/docs"
	@echo "  3. Check system health with: make health-check"
	@echo ""
