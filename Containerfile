# syntax=docker/dockerfile:1
# Lean multi-stage OCI Containerfile compatible with Podman and Docker (rootless, UID 1000).
# Multi-arch: linux/amd64 & linux/arm64 (pure CPU, no PyTorch/CUDA bloat).

FROM ghcr.io/astral-sh/uv:0.7.12 AS uv_bin

# ==============================================================================
# Stage 1: Builder
# ==============================================================================
FROM python:3.12-slim AS builder

COPY --from=uv_bin /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python --no-cache .

# ==============================================================================
# Stage 2: Runtime image (rootless, UID 1000)
# ==============================================================================
FROM python:3.12-slim AS cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 utrader \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash utrader

WORKDIR /app
COPY --from=builder --chown=1000:1000 /app/.venv /app/.venv
COPY --chown=1000:1000 pyproject.toml README.md ./
COPY --chown=1000:1000 configs ./configs

RUN mkdir -p /app/data_cache /app/forecast_cache /app/state /app/reports \
    && chown -R 1000:1000 /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    UTRADER_ROOT=/app \
    MPLCONFIGDIR=/tmp/matplotlib

USER 1000:1000

ENTRYPOINT ["utrader"]
CMD ["doctor"]
