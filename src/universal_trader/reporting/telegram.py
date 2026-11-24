"""Asset-agnostic operational and risk alerting Notifiers (Telegram HTML + Noop)."""

from __future__ import annotations

import html
from pathlib import Path

import httpx

from universal_trader.core.logging import get_logger, register_secret
from universal_trader.core.models import Instrument, PortfolioStepOutcome, StepOutcome
from universal_trader.core.registry import register_notifier

logger = get_logger(__name__)


def _esc(value: object) -> str:
    """HTML-escape dynamic strings so underscores, brackets, or errors never break Telegram."""
    return html.escape(str(value), quote=False)


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
    """Asset- and currency-agnostic Telegram Bot API notifier with HTML escaping & event filtering."""

    name = "telegram"

    def __init__(
        self,
        *,
        bot_token: str = "",
        chat_id: str = "",
        enabled: bool = True,
        notify_on: str = "all",
        daily_summary_hour_utc: int | None = 8,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.bot_token = bot_token
        self.chat_id = str(chat_id)
        register_secret(self.bot_token)
        self.enabled = enabled and bool(self.bot_token) and bool(self.chat_id)
        self.notify_on = notify_on  # "all" | "events"
        self.daily_summary_hour_utc = daily_summary_hour_utc
        self.timeout_seconds = timeout_seconds

    @property
    def _api_base(self) -> str:
        return f"https://api.telegram.org/bot{self.bot_token}"

    def _is_daily_heartbeat(self, ts_hour: int) -> bool:
        return self.daily_summary_hour_utc is not None and ts_hour == self.daily_summary_hour_utc

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

    def should_notify_portfolio_step(self, outcome: PortfolioStepOutcome) -> bool:
        """Decide whether a multi-asset portfolio step should trigger a Telegram message."""
        if outcome.skipped_idempotent:
            return False
        if self.notify_on == "all":
            return True
        if (
            outcome.error
            or outcome.fills
            or outcome.rejected_orders
            or outcome.rebalance_triggered
        ):
            return True
        if outcome.portfolio_after is not None and outcome.portfolio_after.halted:
            return True
        return self._is_daily_heartbeat(outcome.candle_timestamp.hour)

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

    def format_portfolio_step_message(
        self,
        outcome: PortfolioStepOutcome,
        instruments: list[Instrument],
    ) -> str:
        basket = " + ".join(i.base for i in instruments)
        if outcome.error:
            badge = "[ERROR] "
        elif outcome.fills:
            badge = "[TRADE] "
        elif outcome.rejected_orders:
            badge = "[RISK] "
        elif outcome.rebalance_triggered:
            badge = "[SIGNAL] "
        elif self.notify_on == "events":
            badge = "[DAILY] "
        else:
            badge = ""

        lines = [
            f"{badge}<b>{_esc(outcome.config_name)}</b> — <code>{_esc(basket)}</code> ({_esc(outcome.timeframe)})",
            f"Candle UTC: <code>{outcome.candle_timestamp.strftime('%Y-%m-%d %H:%M')}</code>",
        ]
        if outcome.skipped_idempotent:
            lines.append("Status: SKIPPED (idempotent duplicate candle)")
            return "\n".join(lines)
        if outcome.error:
            lines.append(f"Status: <b>ERROR</b> — {_esc(outcome.error)}")
            return "\n".join(lines)

        for inst in instruments:
            sig_lbl = "BULL" if outcome.active_signals.get(inst.id) else "FLAT"
            vol_v = outcome.realized_vols.get(inst.id, 0.0)
            tgt_w = outcome.target_weights.get(inst.id, 0.0)
            aft_w = outcome.current_weights_after.get(inst.id, 0.0)
            lines.append(
                f"• <code>{_esc(inst.base)}</code>: <b>{sig_lbl}</b> | "
                f"vol <code>{vol_v:.0%}</code> | tgt <code>{tgt_w:.0%}</code> | cur <code>{aft_w:.0%}</code>"
            )

        for fl in outcome.fills:
            lines.append(
                f"Order: <b>{_esc(fl.side.value)}</b> <code>{fl.quantity:.6g} {_esc(fl.instrument_id)}</code> "
                f"@ <code>{fl.fill_price:,.4f}</code> (<code>{_esc(fl.status.value)}</code>, fee=<code>{fl.fee:.4f}</code>)"
            )
        for ro in outcome.rejected_orders:
            prefix = "HALTED / " if ro.halted else ""
            lines.append(
                f"Risk {prefix}REJECTED <code>{_esc(ro.instrument_id)}</code> "
                f"(<code>{_esc(ro.guard_name)}</code>): {_esc(ro.reason)}"
            )

        pf = outcome.portfolio_after or outcome.portfolio_before
        if pf is not None:
            lines.append(
                f"Equity: <code>{pf.equity:,.2f} {_esc(pf.quote_currency)}</code> | "
                f"Cash: <code>{pf.cash:,.2f}</code> | DD: <code>{pf.drawdown:.2%}</code>"
            )
        return "\n".join(lines)

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

    def send_test_message(self, label: str = "universal_trader") -> tuple[bool, str]:
        """Send an explicit connectivity test message and return (ok, detail)."""
        msg = (
            f"<b>Test Connessione — {_esc(label)}</b>\n"
            f"Modalità notifiche: <code>{_esc(self.notify_on)}</code> | "
            f"Heartbeat UTC: <code>{self.daily_summary_hour_utc if self.daily_summary_hour_utc is not None else 'off'}</code>\n"
            "Se leggi questo messaggio, il bot Telegram è configurato correttamente."
        )
        return self._send_text(msg)

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
    ) -> None:
        if not self.should_notify_portfolio_step(outcome):
            logger.debug("Skipping quiet Telegram portfolio notification (notify_on=%s).", self.notify_on)
            return
        self._send_text(self.format_portfolio_step_message(outcome, instruments))

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
