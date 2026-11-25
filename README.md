# Universal Trader

**Universal Trader** is a lean, modular, asset-agnostic platform for historical backtesting, walk-forward validation, and automated single-asset & multi-asset trading (Dry-Run, Paper, and Live).

## Key Highlights

- **Single Execution Pipeline:** Backtest, Paper, Dry-Run, and Live all execute the exact same `PortfolioStepRunner` / `TradingStepRunner`, `Strategy`, and `RiskManager` (zero backtest/live drift).
- **Multi-Asset Portfolio Native:** Built-in Active Inverse-Volatility Risk Parity + causal EMA trend & macro regime filter (`configs/bitfinex_portfolio_btc_eth_gold.yaml` for `BTC + ETH + Gold XAUT` on Bitfinex with 0% trading fees).
- **Risk Management & Safety:** 7 composable `RiskGuard`s (market hours, staleness, fee/slippage guard, instrument lot constraints, exposure caps, drawdown circuit breaker, stop-loss) + SELL-before-BUY multi-asset rebalancing.
- **Idempotency, Audit & Telegram Alerts:** WAL-mode SQLite persistence (`runs`, `orders`, `fills`, `equity_snapshots`, `LocalPaperBroker`), advisory execution locks, secret-redacting logs, and customizable Jinja2 HTML Telegram templates (real-time trade alerts + 24h daily summary).
- **Lean & Portable:** Pure-Python lightweight runtime (no GPU/PyTorch required), runs natively via `uv` or rootless inside OCI containers (**Podman** & **Docker**).

## Quickstart (Native with `uv`)

```bash
git clone <repo-url> universal-trader && cd universal-trader
cp .env.example .env
uv sync && uv run utrader doctor
```

### Everyday CLI Commands

```bash
# Environment, storage, plugins, and config diagnostics
uv run utrader doctor

# 6-year multi-asset Bitfinex portfolio backtest (BTC + ETH + Gold XAUT)
uv run utrader backtest --config configs/bitfinex_portfolio_btc_eth_gold.yaml --start 2021-01-01 --end 2026-11-22

# Single atomic trading cycle (add --dry-run for zero broker writes)
uv run utrader trade --config configs/bitfinex_portfolio_btc_eth_gold.yaml --dry-run
uv run utrader trade --config configs/bitfinex_portfolio_btc_eth_gold.yaml

# Preview/test Telegram templates (trade | daily | daily_empty | risk | error)
uv run utrader test-telegram --config configs/bitfinex_portfolio_btc_eth_gold.yaml --scenario trade
uv run utrader test-telegram --config configs/bitfinex_portfolio_btc_eth_gold.yaml --scenario daily

# Operational PnL, drawdown & fills report from SQLite
uv run utrader report --db state/bitfinex_portfolio.db
```

## Containerized Usage (Podman & Docker)

```bash
# Build lean rootless image (linux/amd64 & linux/arm64, UID 1000)
podman build --target cpu -t universal-trader:cpu -f Containerfile .

# Run diagnostics and dry-run via Compose (works identically with `docker compose`)
podman compose run --rm trader doctor
podman compose run --rm trader trade --config configs/bitfinex_portfolio_btc_eth_gold.yaml --dry-run

# Start hourly background scheduler
podman compose --profile scheduler up -d
```

## Development & Testing

```bash
make setup       # uv sync --extra dev
make test        # unit + integration test suite
make lint        # ruff check
make typecheck   # mypy src
```
