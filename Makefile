# Makefile for Factor Model Backtester
# Provides convenient commands for development and testing

.PHONY: help install test lint format clean docs examples benchmark

# Default target
help:
	@echo "Factor Model Backtester - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install     Install dependencies and package"
	@echo "  install-dev Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test        Run all tests"
	@echo "  test-cov    Run tests with coverage report"
	@echo "  benchmark   Run performance benchmarks"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint        Run all linting tools"
	@echo "  format      Format code with black and isort"
	@echo "  type-check  Run type checking with mypy"
	@echo ""
	@echo "Examples:"
	@echo "  example-basic       Run basic backtest example"
	@echo "  example-walkforward Run walk-forward analysis example"
	@echo "  example-custom      Run custom factors example"
	@echo "  notebook            Start Jupyter notebook"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean       Clean cache and temporary files"
	@echo "  docs        Generate documentation"

# Installation targets
install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e .
	pip install pytest pytest-cov black isort flake8 mypy jupyter

# Testing targets
test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ --cov=factor_backtester --cov-report=html --cov-report=term

benchmark:
	python -c "from tests.test_factor_backtester import run_performance_tests; run_performance_tests()"

# Code quality targets
lint: flake8 mypy

flake8:
	flake8 factor_backtester.py tests/ examples/ --max-line-length=100 --ignore=E203,W503

mypy:
	mypy factor_backtester.py --ignore-missing-imports

format:
	black factor_backtester.py tests/ examples/ --line-length=100
	isort factor_backtester.py tests/ examples/ --profile=black

type-check:
	mypy factor_backtester.py --ignore-missing-imports

# Example targets
example-basic:
	python examples/basic_backtest.py

example-walkforward:
	python examples/walk_forward_analysis.py

example-custom:
	python examples/custom_factors.py

notebook:
	jupyter notebook research_notebook.ipynb

# Documentation targets
docs:
	@echo "Generating documentation..."
	@echo "README.md already contains comprehensive documentation"
	@echo "Run 'make notebook' to open the research notebook"

# Maintenance targets
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.log" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf build/
	rm -rf dist/
	rm -rf data_cache/*.pkl
	rm -f *.png

# Development workflow targets
dev-setup: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify installation"

check: format lint test
	@echo "All checks passed! ✅"

# Release targets
build:
	python setup.py sdist bdist_wheel

upload-test:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

upload:
	twine upload dist/*

# Quick start target
quickstart: install example-basic
	@echo ""
	@echo "🚀 Quick start completed!"
	@echo "✅ Dependencies installed"
	@echo "✅ Basic example executed"
	@echo ""
	@echo "Next steps:"
	@echo "  - Run 'make notebook' to explore the research notebook"
	@echo "  - Run 'make example-walkforward' for advanced analysis"
	@echo "  - Run 'make example-custom' to see custom factor examples"

# Continuous integration target
ci: install-dev lint test
	@echo "CI pipeline completed successfully!"

# Docker targets (optional)
docker-build:
	docker build -t factor-backtester .

docker-run:
	docker run -it --rm -v $(PWD):/workspace factor-backtester

# Data management targets
clear-cache:
	rm -rf data_cache/
	mkdir -p data_cache/
	@echo "Data cache cleared"

# Advanced targets
profile:
	python -m cProfile -o profile_output.prof examples/basic_backtest.py
	@echo "Profiling complete. Use 'python -m pstats profile_output.prof' to analyze"

memory-profile:
	python -m memory_profiler examples/basic_backtest.py

# Research targets
research-setup: install-dev
	jupyter contrib nbextension install --user
	jupyter nbextension enable --py widgetsnbextension
	@echo "Research environment setup complete!"

# Validate installation
validate:
	python -c "import factor_backtester; print('✅ Package imported successfully')"
	python -c "from factor_backtester import Backtester, BacktestConfig; print('✅ Main classes imported')"
	python -m pytest tests/test_factor_backtester.py::TestBacktestConfig -v
	@echo "✅ Installation validated!"

# Show environment info
env-info:
	@echo "Environment Information:"
	@echo "Python version: $$(python --version)"
	@echo "Pip version: $$(pip --version)"
	@echo "Working directory: $$(pwd)"
	@echo "Git branch: $$(git branch --show-current 2>/dev/null || echo 'Not a git repository')"
	@echo "Virtual environment: $${VIRTUAL_ENV:-'Not activated'}"

# All-in-one development setup
dev-all: dev-setup validate research-setup
	@echo ""
	@echo "🎉 Complete development environment ready!"
	@echo ""
	@echo "Available commands:"
	@echo "  make test          - Run tests"
	@echo "  make notebook      - Start Jupyter"
	@echo "  make example-basic - Run basic example"
	@echo "  make lint          - Check code quality"
	@echo "  make format        - Format code"
