.PHONY: install run dev test test-cov lint ingest eval eval-real docker-build docker-up docker-down clean

# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------

install:
	pip install -r requirements.txt

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

dev:
	uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

ingest:
	python scripts/ingest_regulations.py --mock-embeddings

ingest-real:
	python scripts/ingest_regulations.py

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test:
	pytest tests/ -q --tb=short

test-cov:
	pytest tests/ --cov=app --cov-report=term-missing --cov-report=html -q

lint:
	ruff check app/ tests/

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

eval:
	python -m eval.run_eval

eval-real:
	python -m eval.run_eval

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean:
	rm -rf data/vectorstore data/compliance.db __pycache__ .pytest_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
