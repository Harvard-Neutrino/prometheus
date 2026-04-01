# Makefile for common development tasks

.PHONY: help install install-dev test test-fast test-unit test-integration test-physics test-cov lint format clean generate-refs docs docs-serve docs-build

help:
	@echo "Available commands:"
	@echo "  make install          - Install package with test dependencies"
	@echo "  make install-dev      - Install package with all dev dependencies"
	@echo "  make test             - Run all tests"
	@echo "  make test-fast        - Run fast tests only (skip slow tests)"
	@echo "  make test-unit        - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make test-physics     - Run physics regression tests"
	@echo "  make test-cov         - Run tests with coverage report"
	@echo "  make lint             - Run code linting"
	@echo "  make format           - Format code with black and isort"
	@echo "  make docs-serve       - Serve documentation locally with live reload"
	@echo "  make docs-build       - Build documentation static site"
	@echo "  make clean            - Clean up build artifacts"
	@echo "  make generate-refs    - Generate reference values for physics tests"

install:
	pip install -e .[test]

install-dev:
	pip install -e .[dev]

test:
	pytest

test-fast:
	pytest -m "not slow"

test-unit:
	pytest -m unit

test-integration:
	pytest -m integration

test-physics:
	pytest -m physics

test-cov:
	pytest --cov=fennel --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	flake8 fennel tests

format:
	black fennel tests
	isort fennel tests

docs-serve:
	@echo "📚 Starting MkDocs development server..."
	@echo "View at: http://127.0.0.1:8000"
	mkdocs serve

docs-build:
	@echo "📚 Building documentation..."
	mkdocs build --clean
	@echo "✅ Documentation built in site/"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf site/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

generate-refs:
	python scripts/generate_reference_values.py
