# Universal Trader

**Universal Trader** is a modular, portable, asset-agnostic platform for probabilistic financial time-series forecasting, realistic historical backtesting, out-of-sample walk-forward optimization, and automated trading (Dry-Run, Paper, and Live).

Powered initially by **Mistral Timestral** (`mistralai/Timestral-{5M,28M,123M,209M}` — 21 quantiles, up to 32,768 context steps, native past-only & known-future covariates, native `NaN` handling) behind an extensible Hexagonal `PluginRegistry`.

## Key Highlights

- **Single Execution Pipeline:** Backtest, Walk-Forward, Paper, Dry-Run, and Live all execute the exact same `TradingStepRunner`, `FeaturePipeline`, `Strategy`, and `RiskManager`.
- **Multi-Asset Native:** Operates seamlessly on 24/7 fractional crypto (`BTC/EUR`) and exchange-scheduled whole-share equities/ETFs (`NVDA`, `VWCE.DE`) via `MarketCalendar` (`exchange_calendars`) and `adj_close` support.
- **Strictly Causal Features:** Past-only covariates (rolling OBI, log returns, realized volatility, backward `merge_asof` macro series) and known-future calendar features with zero lookahead bias.
- **Distribution-First Decisions & Risk:** 6 modular position sizers (including `TailRiskKellySizer` over Timestral's 21 quantiles) + 7 composable `RiskGuard`s (market hours, staleness, round-trip fee+slippage check, instrument constraints, exposure limits, drawdown circuit breaker, dynamic quantile stop-loss).
- **Idempotency & Audit Trail:** WAL-mode SQLite persistence (`runs`, `forecasts`, `orders`, `fills`, `equity_snapshots`, `LocalPaperBroker`) + advisory execution lock + secret-redacting structured logs.
- **Universal Portability:** Runs natively on CPU, NVIDIA CUDA, and Apple Silicon (`uv`), or rootless inside OCI containers (**Podman** & **Docker**).

## Quickstart (Native with `uv`)

```bash
git clone <repo-url> universal-trader && cd universal-trader
cp .env.example .env
uv sync && uv run utrader doctor
```

### Everyday CLI Commands

```bash
# Environment, hardware (CUDA/MPS/CPU), storage, and config diagnostics
uv run utrader doctor

# Probabilistic forecast + multi-quantile fan chart
uv run utrader predict --config configs/btc_eur_timestral_paper.yaml
uv run utrader predict --config configs/nvda_usd_timestral_paper.yaml

# Realistic bar-by-bar historical backtest (default open[t+1] execution)
uv run utrader backtest --config configs/nvda_usd_timestral_paper.yaml --start 2025-01-01 --end 2026-06-01

# Rolling/expanding Walk-Forward optimization with persistent disk forecast cache
uv run utrader walkforward --config configs/walkforward_example.yaml --start 2025-01-01 --end 2025-06-01

# Single atomic trading step (Dry-Run enforces zero mutating broker calls)
uv run utrader trade --config configs/btc_eur_timestral_paper.yaml --dry-run

# Operational PnL, drawdown & ex-post quantile calibration report from SQLite
uv run utrader report --db state/trader.db
```

## Containerized Usage (Podman & Docker)

> **Hardware Note:**
> - **macOS Apple Silicon:** Run natively with `uv` for direct host hardware access (MPS/CPU); containers on macOS run inside a Linux ARM64 VM without Metal GPU passthrough (ideal for isolated CPU execution and Linux parity tests).
> - **Linux + NVIDIA GPU:** Use `--target cuda` with Podman CDI (`--device nvidia.com/gpu=all`) or Docker (`--gpus all`).
> - **Linux SELinux (Fedora/RHEL):** Add `:Z` to bind mounts and `--userns=keep-id` on standalone `podman run` commands.

```bash
# Build lean CPU image (linux/amd64 & linux/arm64, rootless UID 1000)
podman build --target cpu -t universal-trader:cpu -f Containerfile .

# Build NVIDIA CUDA image (linux/amd64)
podman build --target cuda -t universal-trader:cuda -f Containerfile .

# Run diagnostics and dry-run via Compose (works identically with `docker compose`)
podman compose run --rm trader doctor
podman compose run --rm trader trade --config configs/btc_eur_timestral_paper.yaml --dry-run

# NVIDIA GPU execution with Podman CDI
podman run --rm --device nvidia.com/gpu=all \
  --userns=keep-id \
  -v ./configs:/app/configs:ro \
  -v ./state:/app/state:Z \
  -v hf_cache:/app/.hf_cache \
  --env-file .env \
  universal-trader:cuda trade --config configs/btc_eur_timestral_paper.yaml --dry-run
```

## Development & Testing

```bash
make setup       # uv sync --extra dev
make test        # fast unit + integration test suite (no network/weights)
make test-all    # full suite including Timestral-5M inference smoke test
make lint        # ruff check
make typecheck   # mypy src
```
