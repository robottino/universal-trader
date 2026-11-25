"""Interactive Telegram Bot microservice (long-polling) sharing WAL SQLite with the trader."""

from __future__ import annotations

import html
import signal
import time
from datetime import UTC, datetime
from types import FrameType
from typing import Any

import httpx

from universal_trader.brokers.local_paper import LocalPaperBroker
from universal_trader.core.calendar import MarketCalendar
from universal_trader.core.config import AppConfig, RuntimePaths
from universal_trader.core.logging import get_logger, register_secret
from universal_trader.core.models import (
    Instrument,
    PortfolioState,
    PortfolioStepOutcome,
    Quote,
    ensure_utc,
)
from universal_trader.core.protocols import Broker, DataProvider
from universal_trader.engine.portfolio import PortfolioAllocator
from universal_trader.reporting.telegram import TelegramNotifier
from universal_trader.storage.sqlite_store import SQLiteStateStore

logger = get_logger(__name__)


def _esc(value: object) -> str:
    return html.escape(str(value), quote=False)


_BOT_COMMANDS: list[dict[str, str]] = [
    {"command": "status", "description": "Stato live del portafoglio e ultimo ciclo"},
    {"command": "daily", "description": "Riepilogo giornaliero ultime 24h"},
    {"command": "fills", "description": "Ultime 10 operazioni eseguite"},
    {"command": "signals", "description": "Segnali EMA e volatilità attuali"},
    {"command": "halt", "description": "Metti in pausa nuovi acquisti (BUY)"},
    {"command": "resume", "description": "Riattiva l'operatività normale"},
    {"command": "help", "description": "Elenco dei comandi disponibili"},
]


class TelegramBotService:
    """Lightweight long-polling Telegram command listener authorized strictly by chat_id.

    Always receives a read-only (`DryRunBroker`-wrapped) broker so this service can inspect
    live exchange balances and quotes for `/status`, `/daily`, and `/signals`, while physically
    incapable of submitting orders.
    """

    def __init__(
        self,
        *,
        config: AppConfig,
        paths: RuntimePaths,
        store: SQLiteStateStore,
        provider: DataProvider,
        broker: Broker,
        notifier: TelegramNotifier,
        bot_token: str,
        allowed_chat_id: str,
        poll_timeout: int = 25,
        startup_max_age_seconds: int = 120,
    ) -> None:
        if not bot_token or not allowed_chat_id:
            raise ValueError("TelegramBotService requires non-empty bot_token and allowed_chat_id")
        self.config = config
        self.paths = paths
        self.store = store
        self.provider = provider
        self.broker = broker
        self.notifier = notifier
        self.bot_token = bot_token
        self.allowed_chat_id = str(allowed_chat_id).strip()
        register_secret(self.bot_token)
        self.poll_timeout = max(1, int(poll_timeout))
        self.startup_max_age_seconds = max(10, int(startup_max_age_seconds))
        self._stop_requested = False
        self._offset = 0

    @property
    def _api_base(self) -> str:
        return f"https://api.telegram.org/bot{self.bot_token}"

    def request_stop(self, *_: object) -> None:
        logger.info("Stop signal received; shutting down TelegramBotService cleanly.")
        self._stop_requested = True

    # ── Live Portfolio & Quote Helpers ─────────────────────────────────────────────

    def _quote_from_candles(self, inst: Instrument) -> Quote:
        df = self.provider.fetch_ohlcv(instrument=inst, timeframe=self.config.data.timeframe, limit=3)
        col = self.config.data.price_column if self.config.data.price_column in df.columns else "close"
        px = float(df[col].iloc[-1])
        ts = ensure_utc(df.index[-1])
        half_spread = px * (self.config.risk.spread_bps / 20_000.0)
        return Quote(
            instrument_id=inst.id,
            timestamp=ts,
            bid=max(1e-9, px - half_spread),
            ask=px + half_spread,
            last=px,
        )

    def _fetch_quotes_and_portfolio(self) -> tuple[dict[str, Quote], PortfolioState]:
        inner = getattr(self.broker, "_inner", self.broker)
        instruments = list(self.config.instruments)

        if isinstance(inner, LocalPaperBroker):
            quotes = {inst.id: self._quote_from_candles(inst) for inst in instruments}
            inner.set_quotes(quotes)
        else:
            batch_fn = getattr(inner, "get_quotes", None)
            if callable(batch_fn):
                quotes = batch_fn(instruments)
            else:
                quotes = {inst.id: self.broker.get_quote(inst) for inst in instruments}

        multi_fn = getattr(self.broker, "get_multi_portfolio", None)
        if callable(multi_fn):
            pf: PortfolioState = multi_fn(instruments, quotes=quotes)
        else:
            first = instruments[0]
            pf = self.broker.get_portfolio(first, quote=quotes.get(first.id))

        hwm = self.store.get_high_water_mark(self.config.name, default=pf.high_water_mark)
        halted = pf.halted or self.store.is_halted(self.config.name)
        pf = pf.model_copy(
            update={
                "high_water_mark": max(pf.high_water_mark, hwm, pf.equity),
                "halted": halted,
            }
        )
        return quotes, pf

    # ── Command Handlers ───────────────────────────────────────────────────────────

    def _cmd_help(self) -> str:
        lines = [
            f"🤖 <b>Universal Trader Bot — {_esc(self.config.name)}</b>",
            "",
            "<b>Comandi disponibili:</b>",
        ]
        for c in _BOT_COMMANDS:
            lines.append(f"• /{c['command']} — {_esc(c['description'])}")
        return "\n".join(lines)

    def _cmd_status(self) -> str:
        quotes, pf = self._fetch_quotes_and_portfolio()
        last_run = self.store.get_last_run(self.config.name)
        state_lbl = "🛑 <b>IN PAUSA (HALTED)</b>" if pf.halted else "🟢 <b>ATTIVO</b>"
        mode_lbl = self.config.mode.value.upper()
        qccy = pf.quote_currency

        lines = [
            f"📡 <b>Stato Portafoglio — {_esc(self.config.name)}</b>",
            f"Modalità: <code>{_esc(mode_lbl)}</code> · Stato: {state_lbl}",
        ]
        if last_run is not None:
            c_ts = ensure_utc(last_run["candle_timestamp"]).strftime("%Y-%m-%d %H:%M")
            lines.append(
                f"Ultimo ciclo: <code>{c_ts} UTC</code> (<code>{_esc(last_run['status'])}</code>)"
            )
        else:
            lines.append("Ultimo ciclo: <i>nessun ciclo registrato finora</i>")

        lines.extend(
            [
                "",
                "<b>Capitale</b>",
                f"• Equity totale: <code>{pf.equity:,.2f} {_esc(qccy)}</code>",
                f"• Massimo storico (HWM): <code>{pf.high_water_mark:,.2f} {_esc(qccy)}</code>",
                f"• Drawdown attuale: <code>{pf.drawdown:.2%}</code> (max <code>{self.config.risk.max_drawdown_pct:.0%}</code>)",
                "",
                "<b>Posizioni &amp; Prezzi live</b>",
            ]
        )

        for inst in self.config.instruments:
            q = quotes.get(inst.id)
            px = q.last if q is not None else 0.0
            pos = pf.positions.get(inst.id)
            qty = pos.quantity if pos is not None else 0.0
            val = qty * px if px > 0 else (pos.market_value if pos is not None else 0.0)
            w = (val / pf.equity) if pf.equity > 0 else 0.0
            dot = "🟢" if qty > 0 and val > 1.0 else "⚪"
            qty_part = f" · <code>{qty:.6g}</code> {_esc(inst.base)}" if qty > 0 else ""
            lines.append(
                f"{dot} <code>{_esc(inst.base)}</code> (<code>{px:,.2f}</code>): "
                f"<code>{val:,.2f} {_esc(qccy)}</code> (<code>{w:.1%}</code>{qty_part})"
            )

        cash_w = (pf.cash / pf.equity) if pf.equity > 0 else 1.0
        lines.append(
            f"💵 <code>Cash</code>: <code>{pf.cash:,.2f} {_esc(qccy)}</code> (<code>{cash_w:.1%}</code>)"
        )
        return "\n".join(lines)

    def _cmd_daily(self) -> str:
        now = datetime.now(UTC)
        quotes, pf = self._fetch_quotes_and_portfolio()
        weights = {
            inst.id: (pf.position_value(inst.id) / pf.equity) if pf.equity > 0 else 0.0
            for inst in self.config.instruments
        }
        outcome = PortfolioStepOutcome(
            config_name=self.config.name,
            timeframe=self.config.data.timeframe,
            candle_timestamp=now,
            current_weights_before=weights,
            current_weights_after=weights,
            portfolio_before=pf,
            portfolio_after=pf,
        )
        daily_stats = self.store.get_daily_summary_stats(self.config.name, as_of=now)
        return self.notifier.format_daily_summary_message(
            outcome,
            list(self.config.instruments),
            daily_stats=daily_stats,
        )

    def _cmd_fills(self) -> str:
        fills_df = self.store.get_fills(config_name=self.config.name)
        lines = [f"🧾 <b>Ultime operazioni — {_esc(self.config.name)}</b>"]
        if fills_df.empty:
            lines.append("<i>Nessuna operazione registrata nel database.</i>")
            return "\n".join(lines)

        recent = fills_df.tail(10).iloc[::-1]
        qccy = self.config.instruments[0].quote_currency
        lines.append("")
        for _, row in recent.iterrows():
            ts_str = ensure_utc(row["timestamp"]).strftime("%Y-%m-%d %H:%M")
            side = str(row["side"])
            icon = "🟩" if side == "BUY" else "🟥"
            qty = float(row["quantity"])
            px = float(row["fill_price"])
            notional = float(row["notional"])
            status = str(row["status"])
            inst_id = str(row["instrument_id"])
            lines.append(
                f"{icon} <code>{ts_str}</code> <b>{_esc(side)}</b> <code>{qty:.6g}</code> "
                f"<code>{_esc(inst_id)}</code> @ <code>{px:,.2f}</code> "
                f"(~<code>{notional:,.0f} {_esc(qccy)}</code>, <code>{_esc(status)}</code>)"
            )
        return "\n".join(lines)

    def _cmd_signals(self) -> str:
        quotes, pf = self._fetch_quotes_and_portfolio()
        candles_by_id = {
            inst.id: self.provider.fetch_ohlcv(
                instrument=inst,
                timeframe=self.config.data.timeframe,
                limit=self.config.data.lookback_bars,
            )
            for inst in self.config.instruments
        }
        calendar = MarketCalendar(self.config.instrument.calendar)
        allocator = PortfolioAllocator(self.config, calendar)
        decision = allocator.evaluate(candles_by_id=candles_by_id, portfolio=pf, quotes=quotes)

        last_ts = min(ensure_utc(df.index[-1]) for df in candles_by_id.values())
        lines = [
            f"📈 <b>Segnali Strategia — {_esc(self.config.name)}</b>",
            f"Candela: <code>{last_ts.strftime('%Y-%m-%d %H:%M')} UTC</code> ({_esc(self.config.data.timeframe)})",
            "",
        ]
        for inst in self.config.instruments:
            active = decision.active_signals.get(inst.id, False)
            dot = "🟢" if active else "⚪"
            tgt_w = decision.target_weights.get(inst.id, 0.0)
            cur_w = decision.current_weights.get(inst.id, 0.0)
            px = quotes[inst.id].last if inst.id in quotes else 0.0
            detail = decision.reasons.get(inst.id, "")
            lines.append(
                f"{dot} <b>{_esc(inst.base)}</b> (<code>{px:,.2f}</code>) — "
                f"tgt <code>{tgt_w:.0%}</code> · cur <code>{cur_w:.0%}</code>\n"
                f"   <code>{_esc(detail)}</code>"
            )
        rebal_txt = "SÌ" if decision.rebalance_triggered else "NO"
        lines.append(f"\nRibilanciamento richiesto al prossimo giro: <b>{rebal_txt}</b>")
        return "\n".join(lines)

    def _cmd_halt(self) -> str:
        self.store.set_halted(self.config.name, True)
        return (
            f"🛑 <b>Nuovi acquisti messi in PAUSA ({_esc(self.config.name)})</b>\n"
            "• Nessun nuovo ordine <code>BUY</code> verrà aperto.\n"
            "• Eventuali ordini <code>SELL</code> di uscita trend restano attivi per proteggere il capitale.\n"
            "• Invia /resume per riattivare la normale operatività."
        )

    def _cmd_resume(self) -> str:
        self.store.set_halted(self.config.name, False)
        return (
            f"🟢 <b>Operatività RIATTIVATA ({_esc(self.config.name)})</b>\n"
            "• Il blocco manuale è stato rimosso; i nuovi acquisti sono nuovamente abilitati."
        )

    def handle_text(self, chat_id: str, text: str) -> str | None:
        """Authorize chat_id and dispatch a slash command, returning HTML reply (or None if ignored)."""
        if str(chat_id).strip() != self.allowed_chat_id:
            logger.warning("Ignored unauthorized Telegram message from chat_id=%s", chat_id)
            return None

        cleaned = text.strip()
        if not cleaned.startswith("/"):
            return None

        cmd = cleaned.split()[0].split("@")[0].lower()
        try:
            if cmd in {"/start", "/help"}:
                return self._cmd_help()
            if cmd == "/status":
                return self._cmd_status()
            if cmd == "/daily":
                return self._cmd_daily()
            if cmd in {"/fills", "/trades"}:
                return self._cmd_fills()
            if cmd == "/signals":
                return self._cmd_signals()
            if cmd == "/halt":
                return self._cmd_halt()
            if cmd == "/resume":
                return self._cmd_resume()
            return (
                f"❓ Comando sconosciuto: <code>{_esc(cmd)}</code>\n"
                "Invia /help per vedere l'elenco dei comandi."
            )
        except Exception as exc:
            logger.exception("Error executing Telegram command '%s'", cmd)
            return f"🚨 <b>Errore durante {_esc(cmd)}:</b>\n<code>{_esc(f'{type(exc).__name__}: {exc}')}</code>"

    # ── Telegram HTTP Transport & Long-Polling Loop ────────────────────────────────

    def register_commands(self) -> None:
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(
                    f"{self._api_base}/setMyCommands",
                    json={"commands": _BOT_COMMANDS},
                )
                if resp.status_code != 200:
                    logger.warning("Telegram setMyCommands HTTP %s: %s", resp.status_code, resp.text[:200])
        except Exception as exc:
            logger.warning("Telegram setMyCommands failed: %s", exc)

    def _send_reply(self, text: str) -> None:
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(
                    f"{self._api_base}/sendMessage",
                    json={
                        "chat_id": self.allowed_chat_id,
                        "text": text,
                        "parse_mode": "HTML",
                    },
                )
                if resp.status_code != 200:
                    logger.warning("Telegram reply HTTP %s: %s", resp.status_code, resp.text[:200])
        except Exception as exc:
            logger.warning("Telegram reply network error: %s", exc)

    def poll_once(self, *, is_initial_drain: bool = False) -> int:
        """Execute one getUpdates cycle and return number of processed commands."""
        timeout_sec = 0 if is_initial_drain else self.poll_timeout
        req_timeout = float(timeout_sec) + 10.0
        params: dict[str, Any] = {
            "timeout": timeout_sec,
            "allowed_updates": ["message"],
        }
        if self._offset > 0:
            params["offset"] = self._offset

        with httpx.Client(timeout=req_timeout) as client:
            resp = client.get(f"{self._api_base}/getUpdates", params=params)
        if resp.status_code != 200:
            raise RuntimeError(f"Telegram getUpdates HTTP {resp.status_code}: {resp.text[:200]}")

        payload = resp.json()
        updates = payload.get("result") or []
        now_ts = time.time()
        processed = 0

        for upd in updates:
            upd_id = int(upd["update_id"])
            self._offset = max(self._offset, upd_id + 1)
            msg = upd.get("message") or {}
            msg_date = int(msg.get("date") or 0)
            # On initial startup drain, skip stale commands sent long while service was offline
            if is_initial_drain and msg_date > 0 and (now_ts - msg_date) > self.startup_max_age_seconds:
                logger.info("Skipping stale startup Telegram update_id=%d", upd_id)
                continue
            chat = msg.get("chat") or {}
            chat_id = str(chat.get("id", ""))
            text = str(msg.get("text") or "")
            if not text:
                continue
            reply = self.handle_text(chat_id=chat_id, text=text)
            if reply is not None:
                self._send_reply(reply)
                processed += 1

        return processed

    def run_forever(self) -> None:
        """Register commands, drain stale updates safely, and enter long-polling loop."""
        def _sig_handler(signum: int, _frame: FrameType | None) -> None:
            self.request_stop(signum)

        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

        logger.info(
            "Starting TelegramBotService for '%s' (authorized chat_id=%s)",
            self.config.name,
            self.allowed_chat_id,
        )
        self.register_commands()

        try:
            self.poll_once(is_initial_drain=True)
        except Exception as exc:
            logger.warning("Initial Telegram drain warning: %s", exc)

        backoff = 2.0
        while not self._stop_requested:
            try:
                self.poll_once(is_initial_drain=False)
                backoff = 2.0
            except Exception as exc:
                if self._stop_requested:
                    break
                logger.warning("Telegram polling error (%s); retrying in %.1fs", exc, backoff)
                time.sleep(backoff)
                backoff = min(30.0, backoff * 2.0)

        logger.info("TelegramBotService stopped cleanly.")
