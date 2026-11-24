# syntax=docker/dockerfile:1
# Multi-stage OCI Containerfile compatible with Podman and Docker (rootless, UID 1000).
# Targets:
#   - cpu  (default, multi-arch linux/amd64 & linux/arm64, lean PyTorch CPU wheels)
#   - cuda (linux/amd64, NVIDIA CUDA runtime + PyTorch CUDA wheels)

FROM ghcr.io/astral-sh/uv:0.7.12 AS uv_bin

# ==============================================================================
# Stage 1A: Builder for CPU target (multi-arch: amd64 / arm64)
# ==============================================================================
FROM python:3.12-slim AS builder-cpu

COPY --from=uv_bin /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match \
        --no-cache \
        "torch>=2.4" && \
    uv pip install --python /app/.venv/bin/python \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match \
        --no-cache \
        .

# ==============================================================================
# Stage 2A: Runtime image — CPU (default, rootless)
# ==============================================================================
FROM python:3.12-slim AS cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 utrader \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash utrader

WORKDIR /app
COPY --from=builder-cpu --chown=1000:1000 /app/.venv /app/.venv
COPY --chown=1000:1000 pyproject.toml README.md ./
COPY --chown=1000:1000 configs ./configs

RUN mkdir -p /app/.hf_cache /app/data_cache /app/forecast_cache /app/state /app/reports \
    && chown -R 1000:1000 /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    UTRADER_ROOT=/app \
    UTRADER_DEVICE=cpu \
    HF_HOME=/app/.hf_cache \
    MPLCONFIGDIR=/tmp/matplotlib

USER 1000:1000

ENTRYPOINT ["utrader"]
CMD ["doctor"]

# ==============================================================================
# Stage 1B: Builder for NVIDIA CUDA target (linux/amd64)
# ==============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS builder-cuda

COPY --from=uv_bin /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

RUN uv python install 3.12 && \
    uv venv --python 3.12 /app/.venv && \
    uv pip install --python /app/.venv/bin/python \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        --index-strategy unsafe-best-match \
        --no-cache \
        "torch>=2.4" && \
    uv pip install --python /app/.venv/bin/python \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        --index-strategy unsafe-best-match \
        --no-cache \
        .

# ==============================================================================
# Stage 2B: Runtime image — NVIDIA CUDA (linux/amd64, rootless)
# ==============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 utrader \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash utrader

WORKDIR /app
COPY --from=builder-cuda --chown=1000:1000 /opt/uv-python /opt/uv-python
COPY --from=builder-cuda --chown=1000:1000 /app/.venv /app/.venv
COPY --chown=1000:1000 pyproject.toml README.md ./
COPY --chown=1000:1000 configs ./configs

RUN mkdir -p /app/.hf_cache /app/data_cache /app/forecast_cache /app/state /app/reports \
    && chown -R 1000:1000 /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    UTRADER_ROOT=/app \
    UTRADER_DEVICE=auto \
    HF_HOME=/app/.hf_cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

USER 1000:1000

ENTRYPOINT ["utrader"]
CMD ["doctor"]
