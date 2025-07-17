.PHONY: help install install-dev clean format lint test run-examples

help:
	@echo "Available commands:"
	@echo "  make install       Install the package with uv"
	@echo "  make install-dev   Install with development dependencies"
	@echo "  make clean         Clean up cache and build files"
	@echo "  make format        Format code with black and ruff"
	@echo "  make lint          Run linting checks"
	@echo "  make test          Run tests"
	@echo "  make run-examples  Run all example scripts"

install:
	uv venv
	uv pip install -e .

install-dev:
	uv venv
	uv pip install -e ".[dev]"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/

format:
	uv run black src/ examples/
	uv run ruff check --fix src/ examples/

lint:
	uv run ruff check src/ examples/
	uv run mypy src/

test:
	uv run pytest tests/ -v

run-examples:
	@echo "Running text generation examples..."
	uv run python examples/text_generation.py
	@echo "\nRunning image analysis examples..."
	uv run python examples/image_analysis.py
	@echo "\nRunning audio processing examples..."
	uv run python examples/audio_processing.py
	@echo "\nRunning multimodal pipeline examples..."
	uv run python examples/multimodal_pipeline.py