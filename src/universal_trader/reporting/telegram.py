"""Asset-agnostic operational and risk alerting Notifiers (Telegram HTML + Jinja2 + Noop)."""

from __future__ import annotations

import html
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from jinja2 import Environment, StrictUndefined

from universal_trader.core.logging import get_logger, register_secret
from universal_trader.core.models import (
    AssetClass,
    FillResult,
    Instrument,
    OrderSide,
    OrderStatus,
    PortfolioState,
    PortfolioStepOutcome,
    Position,
    RejectedOrder,
    StepOutcome,
    ensure_utc,
)
from universal_trader.core.registry import register_notifier

logger = get_logger(__name__)


def _esc(value: object) -> str:
    """HTML-escape dynamic strings so underscores, brackets, or errors never break Telegram."""
    return html.escape(str(value), quote=False)


def build_sample_portfolio_outcome(
    *,
    config_name: str,
    instruments: list[Instrument],
    timeframe: str = "1h",
    scenario: str = "trade",
) -> tuple[PortfolioStepOutcome, dict[str, Any]]:
    """Build a realistic sample PortfolioStepOutcome + 24h daily_stats for template preview/testing."""
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    if not instruments:
        instruments = [
            Instrument(id="BTC/USD", asset_class=AssetClass.CRYPTO, base="BTC", quote_currency="USD"),
            Instrument(id="ETH/USD", asset_class=AssetClass.CRYPTO, base="ETH", quote_currency="USD"),
            Instrument(id="XAUT/USD", asset_class=AssetClass.CRYPTO, base="XAUT", quote_currency="USD"),
        ]

    qccy = instruments[0].quote_currency
    sample_vols = [0.48, 0.66, 0.14]
    vols = {inst.id: sample_vols[i % len(sample_vols)] for i, inst in enumerate(instruments)}

    active_inst = instruments[-1]
    active_signals = {inst.id: (inst.id == active_inst.id) for inst in instruments}
    target_weights = {inst.id: (0.60 if inst.id == active_inst.id else 0.0) for inst in instruments}
    weights_before = {inst.id: (0.0 if scenario == "trade" else target_weights[inst.id]) for inst in instruments}
    weights_after = dict(target_weights)

    positions: dict[str, Position] = {}
    if scenario != "error":
        positions[active_inst.id] = Position(
            instrument_id=active_inst.id,
            quantity=1.662,
            avg_entry_price=3610.0,
            market_price=3610.10,
        )

    pf = PortfolioState(
        timestamp=now,
        quote_currency=qccy,
        cash=4000.0,
        positions=positions,
        equity=10000.0,
        high_water_mark=10250.0,
        halted=(scenario == "risk"),
    )

    fills: list[FillResult] = []
    rejected: list[RejectedOrder] = []
    error_msg: str | None = None
    rebalance = False

    if scenario == "trade":
        rebalance = True
        fills.append(
            FillResult(
                order_id="sample-001",
                instrument_id=active_inst.id,
                timestamp=now,
                side=OrderSide.BUY,
                quantity=1.662,
                fill_price=3610.40,
                notional=6000.48,
                fee=0.0,
                slippage=1.80,
                status=OrderStatus.FILLED,
            )
        )
    elif scenario == "risk":
        rebalance = True
        rejected.append(
            RejectedOrder(
                instrument_id=instruments[0].id,
                timestamp=now,
                side=OrderSide.BUY,
                guard_name="DrawdownCircuitBreaker",
                reason="Drawdown 40.2% >= max 40.0%. Only closing orders permitted.",
                halted=True,
            )
        )
    elif scenario == "error":
        error_msg = "BitfinexBrokerError: HTTP 503 Service Unavailable (timeout)"
    elif scenario not in {"daily", "daily_empty"}:
        raise ValueError(
            f"Unsupported test scenario '{scenario}'. Use: trade | daily | daily_empty | risk | error"
        )

    outcome = PortfolioStepOutcome(
        config_name=config_name,
        timeframe=timeframe,
        candle_timestamp=now,
        skipped_idempotent=False,
        active_signals=active_signals,
        realized_vols=vols,
        target_weights=target_weights,
        current_weights_before=weights_before,
        current_weights_after=weights_after,
        rebalance_triggered=rebalance,
        fills=fills,
        rejected_orders=rejected,
        portfolio_before=pf,
        portfolio_after=pf,
        error=error_msg,
    )

    daily_fills: list[dict[str, Any]] = []
    if scenario == "daily":
        daily_fills = [
            {
                "timestamp": (now - timedelta(hours=14)).isoformat(),
                "instrument_id": instruments[0].id,
                "side": "SELL",
                "quantity": 0.045,
                "fill_price": 60250.0,
                "notional": 2711.25,
                "fee": 0.0,
                "status": "FILLED",
            },
            {
                "timestamp": (now - timedelta(hours=6)).isoformat(),
                "instrument_id": active_inst.id,
                "side": "BUY",
                "quantity": 1.662,
                "fill_price": 3610.40,
                "notional": 6000.48,
                "fee": 0.0,
                "status": "FILLED",
            },
        ]

    daily_stats: dict[str, Any] = {
        "period_start": now - timedelta(hours=24),
        "period_end": now,
        "start_equity": 9850.0,
        "fills": daily_fills,
    }
    return outcome, daily_stats


@register_notifier("noop")
class NoopNotifier:
    """Silent notifier used when notifications are disabled."""

    name = "noop"

    def __init__(self, **_: object) -> None:
        pass

    def notify_step(
        self,
        outcome: StepOutcome,
        instrument: Instrument,
        chart_path: Path | None = None,
    ) -> None:
        return None

    def notify_portfolio_step(
        self,
        outcome: PortfolioStepOutcome,
        instruments: list[Instrument],
        daily_stats: dict[str, Any] | None = None,
    ) -> None:
        return None

    def notify_alert(
        self,
        title: str,
        message: str,
        instrument: Instrument | None = None,
    ) -> None:
        return None


@register_notifier("telegram")
class TelegramNotifier:
    """Asset- and currency-agnostic Telegram Bot API notifier backed by external Jinja2 HTML templates."""

    name = "telegram"

    def __init__(
        self,
        *,
        bot_token: str = "",
        chat_id: str = "",
        enabled: bool = True,
        notify_on: str = "all",
        daily_summary_hour_utc: int | None = 8,
        template_path: str | Path = "configs/templates/telegram_portfolio.html.j2",
        daily_template_path: str | Path = "configs/templates/telegram_daily.html.j2",
        timeout_seconds: float = 15.0,
    ) -> None:
        self.bot_token = bot_token
        self.chat_id = str(chat_id)
        register_secret(self.bot_token)
        self.enabled = enabled and bool(self.bot_token) and bool(self.chat_id)
        self.notify_on = notify_on  # "all" | "events"
        self.daily_summary_hour_utc = daily_summary_hour_utc
        self.template_path = Path(template_path)
        self.daily_template_path = Path(daily_template_path)
        self.timeout_seconds = timeout_seconds
        self._jinja_env = Environment(
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
        )

    @property
    def _api_base(self) -> str:
        return f"https://api.telegram.org/bot{self.bot_token}"

    def _is_daily_heartbeat(self, ts_hour: int) -> bool:
        return self.daily_summary_hour_utc is not None and ts_hour == self.daily_summary_hour_utc

    def _render_template(self, path: Path, context: dict[str, Any]) -> str:
        if not path.exists():
            raise FileNotFoundError(f"Telegram template file not found: {path}")
        source = path.read_text(encoding="utf-8")
        template = self._jinja_env.from_string(source)
        return template.render(**context).strip()

    def should_notify_step(self, outcome: StepOutcome) -> bool:
        """Decide whether a single-asset step should trigger a Telegram message."""
        if outcome.skipped_idempotent:
            return False
        if self.notify_on == "all":
            return True
        if outcome.error or outcome.fill is not None or outcome.rejected_order is not None:
            return True
        if outcome.portfolio_after is not None and outcome.portfolio_after.halted:
            return True
        return self._is_daily_heartbeat(outcome.candle_timestamp.hour)

    def should_notify_portfolio_event(self, outcome: PortfolioStepOutcome) -> bool:
        """True when an operational movement, risk rejection, or error should be sent immediately."""
        if outcome.skipped_idempotent:
            return False
        if outcome.error or outcome.fills or outcome.rejected_orders:
            return True
        return self.notify_on == "all" and not self._is_daily_heartbeat(outcome.candle_timestamp.hour)

    def should_notify_portfolio_daily(self, outcome: PortfolioStepOutcome) -> bool:
        """True when the scheduled daily summary digest should be sent."""
        if outcome.skipped_idempotent:
            return False
        return self._is_daily_heartbeat(outcome.candle_timestamp.hour)

    def should_notify_portfolio_step(self, outcome: PortfolioStepOutcome) -> bool:
        """True if either an event notification or the daily summary should be sent for this step."""
        return self.should_notify_portfolio_event(outcome) or self.should_notify_portfolio_daily(outcome)

    def format_step_message(self, outcome: StepOutcome, instrument: Instrument) -> str:
        tag = ""
        if self.notify_on == "events" and not (outcome.fill or outcome.rejected_order or outcome.error):
            tag = "[DAILY] "
        lines = [
            f"{tag}<b>{_esc(outcome.config_name)}</b> — <code>{_esc(instrument.id)}</code> ({_esc(outcome.timeframe)})",
            f"Candle UTC: <code>{outcome.candle_timestamp.strftime('%Y-%m-%d %H:%M')}</code>",
        ]
        if outcome.skipped_idempotent:
            lines.append("Status: SKIPPED (idempotent duplicate candle)")
            return "\n".join(lines)
        if outcome.error:
            lines.append(f"Status: <b>ERROR</b> — {_esc(outcome.error)}")
            return "\n".join(lines)

        if outcome.forecast is not None and outcome.intent is not None:
            sig = outcome.intent.signal
            lines.append(
                f"Spot: <code>{sig.current_price:,.4f} {_esc(instrument.quote_currency)}</code> | "
                f"H={outcome.forecast.horizon} Median: <code>{sig.target_price_median:,.4f}</code> "
                f"({sig.expected_return:+.2%})"
            )
            lines.append(
                f"Spread(q10–q90): <code>{sig.uncertainty_spread:.2%}</code> | "
                f"Tail(q05): <code>{sig.downside_tail_return:+.2%}</code> | "
                f"P(&gt;spot): <code>{sig.prob_positive:.1%}</code>"
            )

        if outcome.fill is not None:
            fl = outcome.fill
            lines.append(
                f"Order: <b>{_esc(fl.side.value)}</b> <code>{fl.quantity:.6g} {_esc(instrument.base)}</code> "
                f"@ <code>{fl.fill_price:,.4f} {_esc(instrument.quote_currency)}</code> "
                f"(status=<code>{_esc(fl.status.value)}</code>, fee=<code>{fl.fee:.4f}</code>)"
            )
        elif outcome.rejected_order is not None:
            ro = outcome.rejected_order
            prefix = "HALTED / " if ro.halted else ""
            lines.append(
                f"Risk {prefix}REJECTED (<code>{_esc(ro.guard_name)}</code>): {_esc(ro.reason)}"
            )
        elif outcome.intent is not None:
            lines.append(f"Action: <b>{_esc(outcome.intent.side.value)}</b> — {_esc(outcome.intent.reason)}")

        pf = outcome.portfolio_after or outcome.portfolio_before
        if pf is not None:
            lines.append(
                f"Equity: <code>{pf.equity:,.2f} {_esc(pf.quote_currency)}</code> | "
                f"Cash: <code>{pf.cash:,.2f}</code> | DD: <code>{pf.drawdown:.2%}</code>"
            )
        return "\n".join(lines)

    def _build_portfolio_context(
        self,
        outcome: PortfolioStepOutcome,
        instruments: list[Instrument],
        *,
        is_test: bool = False,
    ) -> dict[str, Any]:
        if outcome.error:
            badge = "ERROR"
        elif outcome.fills:
            badge = "TRADE"
        elif outcome.rejected_orders:
            badge = "RISK"
        elif outcome.rebalance_triggered:
            badge = "SIGNAL"
        else:
            badge = "STEP"

        assets = [
            {
                "id": inst.id,
                "base": inst.base,
                "active": bool(outcome.active_signals.get(inst.id, False)),
                "signal": "BULL" if outcome.active_signals.get(inst.id) else "FLAT",
                "vol_pct": f"{outcome.realized_vols.get(inst.id, 0.0):.0%}",
                "target_pct": f"{outcome.target_weights.get(inst.id, 0.0):.0%}",
                "before_pct": f"{outcome.current_weights_before.get(inst.id, 0.0):.0%}",
                "after_pct": f"{outcome.current_weights_after.get(inst.id, 0.0):.0%}",
            }
            for inst in instruments
        ]

        fills = [
            {
                "side": fl.side.value,
                "quantity": f"{fl.quantity:.6g}",
                "instrument_id": fl.instrument_id,
                "fill_price": f"{fl.fill_price:,.4f}",
                "notional": f"{fl.notional:,.2f}",
                "fee": f"{fl.fee:.4f}",
                "status": fl.status.value,
            }
            for fl in outcome.fills
        ]

        rejected_orders = [
            {
                "instrument_id": ro.instrument_id,
                "guard_name": ro.guard_name,
                "reason": ro.reason,
                "halted": ro.halted,
            }
            for ro in outcome.rejected_orders
        ]

        pf = outcome.portfolio_after or outcome.portfolio_before
        portfolio_ctx: dict[str, Any] | None = None
        if pf is not None:
            cash_pct = (pf.cash / pf.equity) if pf.equity > 0 else 1.0
            portfolio_ctx = {
                "equity": f"{pf.equity:,.2f}",
                "cash": f"{pf.cash:,.2f}",
                "cash_pct": f"{cash_pct:.0%}",
                "drawdown_pct": f"{pf.drawdown:.2%}",
                "quote_currency": pf.quote_currency,
                "halted": pf.halted,
            }

        return {
            "badge": badge,
            "is_test": is_test,
            "config_name": outcome.config_name,
            "basket": " + ".join(i.base for i in instruments),
            "timeframe": outcome.timeframe,
            "candle_utc": outcome.candle_timestamp.strftime("%Y-%m-%d %H:%M"),
            "skipped_idempotent": outcome.skipped_idempotent,
            "error": outcome.error,
            "assets": assets,
            "fills": fills,
            "rejected_orders": rejected_orders,
            "portfolio": portfolio_ctx,
        }

    def format_portfolio_step_message(
        self,
        outcome: PortfolioStepOutcome,
        instruments: list[Instrument],
        *,
        is_test: bool = False,
    ) -> str:
        """Render operational event message strictly from template_path (fail-fast if missing)."""
        ctx = self._build_portfolio_context(outcome, instruments, is_test=is_test)
        return self._render_template(self.template_path, ctx)

    def format_daily_summary_message(
        self,
        outcome: PortfolioStepOutcome,
        instruments: list[Instrument],
        daily_stats: dict[str, Any] | None = None,
        *,
        is_test: bool = False,
    ) -> str:
        """Render dedicated 24h daily digest strictly from daily_template_path (fail-fast if missing)."""
        stats = daily_stats or {}
        end_dt = ensure_utc(stats.get("period_end") or outcome.candle_timestamp)
        start_dt = ensure_utc(stats.get("period_start") or (end_dt - timedelta(hours=24)))

        pf = outcome.portfolio_after or outcome.portfolio_before
        end_equity = pf.equity if pf is not None else 0.0
        start_equity_raw = stats.get("start_equity")
        start_equity = float(start_equity_raw) if start_equity_raw is not None else end_equity
        delta_abs = end_equity - start_equity
        delta_pct = (delta_abs / start_equity) if start_equity > 0 else 0.0

        quote_ccy = pf.quote_currency if pf is not None else instruments[0].quote_currency
        cash = pf.cash if pf is not None else 0.0
        cash_pct = (cash / end_equity) if end_equity > 0 else 1.0
        drawdown_pct = pf.drawdown if pf is not None else 0.0
        halted = pf.halted if pf is not None else False

        allocations: list[dict[str, Any]] = []
        for inst in instruments:
            pos = pf.positions.get(inst.id) if pf is not None else None
            qty = pos.quantity if pos is not None else 0.0
            val = pos.market_value if pos is not None else 0.0
            weight = (val / end_equity) if end_equity > 0 else 0.0
            allocations.append(
                {
                    "id": inst.id,
                    "base": inst.base,
                    "invested": qty > 0 and val > 0,
                    "quantity": f"{qty:.6g}",
                    "value": f"{val:,.2f}",
                    "weight_pct": f"{weight:.1%}",
                }
            )

        raw_fills: list[dict[str, Any]] = list(stats.get("fills") or [])
        daily_fills: list[dict[str, Any]] = []
        for rf in raw_fills:
            ts_val = ensure_utc(rf["timestamp"])
            daily_fills.append(
                {
                    "time_utc": ts_val.strftime("%H:%M"),
                    "side": str(rf["side"]),
                    "instrument_id": str(rf["instrument_id"]),
                    "quantity": f"{float(rf['quantity']):.6g}",
                    "fill_price": f"{float(rf['fill_price']):,.4f}",
                    "notional": f"{float(rf['notional']):,.2f}",
                    "fee": f"{float(rf.get('fee', 0.0)):.4f}",
                    "status": str(rf.get("status", "FILLED")),
                }
            )

        ctx: dict[str, Any] = {
            "is_test": is_test,
            "config_name": outcome.config_name,
            "basket": " + ".join(i.base for i in instruments),
            "period_start_utc": start_dt.strftime("%Y-%m-%d %H:%M"),
            "period_end_utc": end_dt.strftime("%Y-%m-%d %H:%M"),
            "quote_currency": quote_ccy,
            "start_equity": f"{start_equity:,.2f}",
            "end_equity": f"{end_equity:,.2f}",
            "delta_abs": f"{delta_abs:+,.2f}",
            "delta_pct": f"{delta_pct:+.2%}",
            "drawdown_pct": f"{drawdown_pct:.2%}",
            "halted": halted,
            "cash": f"{cash:,.2f}",
            "cash_pct": f"{cash_pct:.1%}",
            "allocations": allocations,
            "daily_fills": daily_fills,
        }
        return self._render_template(self.daily_template_path, ctx)

    def _send_text(self, text: str) -> tuple[bool, str]:
        if not self.enabled:
            logger.debug("TelegramNotifier disabled or unconfigured; skipping message.")
            return False, "TelegramNotifier is disabled or missing bot_token/chat_id"
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(
                    f"{self._api_base}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": "HTML",
                    },
                )
                if resp.status_code != 200:
                    err_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    logger.warning("Telegram sendMessage failed (%s)", err_msg)
                    return False, err_msg
                return True, "OK"
        except Exception as exc:
            logger.warning("Telegram sendMessage exception: %s", exc)
            return False, str(exc)

    def _send_photo(self, photo_path: Path, caption: str) -> None:
        if not self.enabled:
            return
        try:
            with photo_path.open("rb") as fh, httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(
                    f"{self._api_base}/sendPhoto",
                    data={"chat_id": self.chat_id, "caption": caption[:1024], "parse_mode": "HTML"},
                    files={"photo": (photo_path.name, fh, "image/png")},
                )
                if resp.status_code != 200:
                    logger.warning("Telegram sendPhoto failed (%s); falling back to text.", resp.status_code)
                    self._send_text(caption)
        except Exception as exc:
            logger.warning("Telegram sendPhoto exception: %s", exc)
            self._send_text(caption)

    def send_portfolio_test_message(
        self,
        *,
        config_name: str,
        instruments: list[Instrument],
        timeframe: str = "1h",
        scenario: str = "trade",
        dry_run: bool = False,
    ) -> tuple[bool, str, str]:
        """Render the appropriate external template (event or daily) and optionally send it to Telegram."""
        sample, daily_stats = build_sample_portfolio_outcome(
            config_name=config_name,
            instruments=instruments,
            timeframe=timeframe,
            scenario=scenario,
        )
        if scenario in {"daily", "daily_empty"}:
            rendered = self.format_daily_summary_message(
                sample,
                instruments,
                daily_stats=daily_stats,
                is_test=True,
            )
        else:
            rendered = self.format_portfolio_step_message(sample, instruments, is_test=True)

        if dry_run:
            return True, "DRY-RUN", rendered
        ok, detail = self._send_text(rendered)
        return ok, detail, rendered

    def notify_step(
        self,
        outcome: StepOutcome,
        instrument: Instrument,
        chart_path: Path | None = None,
    ) -> None:
        if not self.should_notify_step(outcome):
            logger.debug("Skipping quiet Telegram step notification (notify_on=%s).", self.notify_on)
            return
        msg = self.format_step_message(outcome, instrument)
        if chart_path is not None and chart_path.exists():
            self._send_photo(chart_path, msg)
        else:
            self._send_text(msg)

    def notify_portfolio_step(
        self,
        outcome: PortfolioStepOutcome,
        instruments: list[Instrument],
        daily_stats: dict[str, Any] | None = None,
    ) -> None:
        if self.should_notify_portfolio_event(outcome):
            msg = self.format_portfolio_step_message(outcome, instruments)
            self._send_text(msg)
        if self.should_notify_portfolio_daily(outcome):
            daily_msg = self.format_daily_summary_message(outcome, instruments, daily_stats=daily_stats)
            self._send_text(daily_msg)

    def notify_alert(
        self,
        title: str,
        message: str,
        instrument: Instrument | None = None,
    ) -> None:
        header = f"ALERT: <b>{_esc(title)}</b>"
        if instrument is not None:
            header += f" (<code>{_esc(instrument.id)}</code>)"
        self._send_text(f"{header}\n{_esc(message)}")
