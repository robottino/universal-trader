"""Asset-agnostic operational and risk alerting Notifiers (Telegram + Noop)."""

from __future__ import annotations

from pathlib import Path

import httpx

from universal_trader.core.logging import get_logger, register_secret
from universal_trader.core.models import Instrument, PortfolioStepOutcome, StepOutcome
from universal_trader.core.registry import register_notifier

logger = get_logger(__name__)


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
    """Asset- and currency-agnostic Telegram Bot API notifier."""

    name = "telegram"

    def __init__(
        self,
        *,
        bot_token: str = "",
        chat_id: str = "",
        enabled: bool = True,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.bot_token = bot_token
        self.chat_id = str(chat_id)
        register_secret(self.bot_token)
        self.enabled = enabled and bool(self.bot_token) and bool(self.chat_id)
        self.timeout_seconds = timeout_seconds

    @property
    def _api_base(self) -> str:
        return f"https://api.telegram.org/bot{self.bot_token}"

    def format_step_message(self, outcome: StepOutcome, instrument: Instrument) -> str:
        lines = [
            f"*{outcome.config_name}* — `{instrument.id}` ({outcome.timeframe})",
            f"Candle UTC: `{outcome.candle_timestamp.strftime('%Y-%m-%d %H:%M')}`",
        ]
        if outcome.skipped_idempotent:
            lines.append("Status: SKIPPED (idempotent duplicate candle)")
            return "\n".join(lines)
        if outcome.error:
            lines.append(f"Status: ERROR — {outcome.error}")
            return "\n".join(lines)

        if outcome.forecast is not None and outcome.intent is not None:
            sig = outcome.intent.signal
            lines.append(
                f"Spot: `{sig.current_price:,.4f} {instrument.quote_currency}` | "
                f"H={outcome.forecast.horizon} Median: `{sig.target_price_median:,.4f}` "
                f"({sig.expected_return:+.2%})"
            )
            lines.append(
                f"Spread(q10–q90): `{sig.uncertainty_spread:.2%}` | "
                f"Tail(q05): `{sig.downside_tail_return:+.2%}` | "
                f"P(>spot): `{sig.prob_positive:.1%}`"
            )

        if outcome.fill is not None:
            fl = outcome.fill
            lines.append(
                f"Order: *{fl.side.value}* `{fl.quantity:.6g} {instrument.base}` "
                f"@ `{fl.fill_price:,.4f} {instrument.quote_currency}` "
                f"(status=`{fl.status.value}`, fee=`{fl.fee:.4f}`)"
            )
        elif outcome.rejected_order is not None:
            ro = outcome.rejected_order
            prefix = "HALTED / " if ro.halted else ""
            lines.append(
                f"Risk {prefix}REJECTED (`{ro.guard_name}`): {ro.reason}"
            )
        elif outcome.intent is not None:
            lines.append(f"Action: *{outcome.intent.side.value}* — {outcome.intent.reason}")

        pf = outcome.portfolio_after or outcome.portfolio_before
        if pf is not None:
            lines.append(
                f"Equity: `{pf.equity:,.2f} {pf.quote_currency}` | "
                f"Cash: `{pf.cash:,.2f}` | DD: `{pf.drawdown:.2%}`"
            )
        return "\n".join(lines)

    def _send_text(self, text: str) -> None:
        if not self.enabled:
            logger.debug("TelegramNotifier disabled or unconfigured; skipping message.")
            return
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(
                    f"{self._api_base}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": "Markdown",
                    },
                )
                if resp.status_code != 200:
                    logger.warning("Telegram sendMessage failed (%s): %s", resp.status_code, resp.text[:200])
        except Exception as exc:
            logger.warning("Telegram sendMessage exception: %s", exc)

    def _send_photo(self, photo_path: Path, caption: str) -> None:
        if not self.enabled:
            return
        try:
            with photo_path.open("rb") as fh, httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(
                    f"{self._api_base}/sendPhoto",
                    data={"chat_id": self.chat_id, "caption": caption[:1024], "parse_mode": "Markdown"},
                    files={"photo": (photo_path.name, fh, "image/png")},
                )
                if resp.status_code != 200:
                    logger.warning("Telegram sendPhoto failed (%s); falling back to text.", resp.status_code)
                    self._send_text(caption)
        except Exception as exc:
            logger.warning("Telegram sendPhoto exception: %s", exc)
            self._send_text(caption)

    def notify_step(
        self,
        outcome: StepOutcome,
        instrument: Instrument,
        chart_path: Path | None = None,
    ) -> None:
        msg = self.format_step_message(outcome, instrument)
        if chart_path is not None and chart_path.exists():
            self._send_photo(chart_path, msg)
        else:
            self._send_text(msg)

    def format_portfolio_step_message(
        self,
        outcome: PortfolioStepOutcome,
        instruments: list[Instrument],
    ) -> str:
        basket = " + ".join(i.base for i in instruments)
        lines = [
            f"*{outcome.config_name}* — `{basket}` ({outcome.timeframe})",
            f"Candle UTC: `{outcome.candle_timestamp.strftime('%Y-%m-%d %H:%M')}`",
        ]
        if outcome.skipped_idempotent:
            lines.append("Status: SKIPPED (idempotent duplicate candle)")
            return "\n".join(lines)
        if outcome.error:
            lines.append(f"Status: ERROR — {outcome.error}")
            return "\n".join(lines)

        for inst in instruments:
            sig_lbl = "BULL" if outcome.active_signals.get(inst.id) else "FLAT"
            vol_v = outcome.realized_vols.get(inst.id, 0.0)
            tgt_w = outcome.target_weights.get(inst.id, 0.0)
            aft_w = outcome.current_weights_after.get(inst.id, 0.0)
            lines.append(f"• `{inst.base}`: {sig_lbl} | vol `{vol_v:.0%}` | tgt `{tgt_w:.0%}` | cur `{aft_w:.0%}`")

        for fl in outcome.fills:
            lines.append(
                f"Order: *{fl.side.value}* `{fl.quantity:.6g} {fl.instrument_id}` "
                f"@ `{fl.fill_price:,.4f}` (`{fl.status.value}`, fee=`{fl.fee:.4f}`)"
            )
        for ro in outcome.rejected_orders:
            prefix = "HALTED / " if ro.halted else ""
            lines.append(f"Risk {prefix}REJECTED `{ro.instrument_id}` (`{ro.guard_name}`): {ro.reason}")

        pf = outcome.portfolio_after or outcome.portfolio_before
        if pf is not None:
            lines.append(
                f"Equity: `{pf.equity:,.2f} {pf.quote_currency}` | "
                f"Cash: `{pf.cash:,.2f}` | DD: `{pf.drawdown:.2%}`"
            )
        return "\n".join(lines)

    def notify_portfolio_step(
        self,
        outcome: PortfolioStepOutcome,
        instruments: list[Instrument],
    ) -> None:
        self._send_text(self.format_portfolio_step_message(outcome, instruments))

    def notify_alert(
        self,
        title: str,
        message: str,
        instrument: Instrument | None = None,
    ) -> None:
        header = f"ALERT: *{title}*"
        if instrument is not None:
            header += f" (`{instrument.id}`)"
        self._send_text(f"{header}\n{message}")
