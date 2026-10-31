.PHONY: install lint format typecheck test test-all benchmark sandbox-image

install:
	uv sync

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck:
	uv run mypy

test:
	uv run pytest -m "not slow and not live_llm and not network"

test-all:
	uv run pytest

benchmark:
	uv run python -m strategy_court.benchmark.suite

sandbox-image:
	docker build -t strategy-court-sandbox:latest -f src/strategy_court/sandbox/Dockerfile .
