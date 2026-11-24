.PHONY: setup test test-all lint typecheck doctor build-cpu build-cuda compose-doctor clean

CONTAINER_ENGINE ?= $(shell command -v podman >/dev/null 2>&1 && echo podman || echo docker)

setup:
	uv sync --extra dev

test:
	uv run pytest -m "not slow and not network"

test-all:
	uv run pytest

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src

doctor:
	uv run utrader doctor

build-cpu:
	$(CONTAINER_ENGINE) build --target cpu -t universal-trader:cpu -f Containerfile .

build-cuda:
	$(CONTAINER_ENGINE) build --target cuda -t universal-trader:cuda -f Containerfile .

compose-doctor:
	$(CONTAINER_ENGINE) compose run --rm trader doctor

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
