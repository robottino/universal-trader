.PHONY: setup test lint typecheck doctor build compose-doctor clean

CONTAINER_ENGINE ?= $(shell command -v podman >/dev/null 2>&1 && echo podman || echo docker)

setup:
	uv sync --extra dev

test:
	uv run pytest -m "not slow and not network"

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src

doctor:
	uv run utrader doctor

build:
	$(CONTAINER_ENGINE) build --target cpu -t universal-trader:cpu -f Containerfile .

compose-doctor:
	$(CONTAINER_ENGINE) compose run --rm trader doctor

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
