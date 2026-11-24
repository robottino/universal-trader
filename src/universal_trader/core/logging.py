"""Structured logging with automatic secret redaction."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import UTC, datetime
from threading import Lock
from typing import Any

_SECRETS_LOCK = Lock()
_REGISTERED_SECRETS: set[str] = set()

_SENSITIVE_ENV_FRAGMENTS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "PASSWD", "CREDENTIAL")
_PATTERN_REDACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"bot\d+:[A-Za-z0-9_-]{16,}"), "bot***REDACTED***"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE), r"\1***REDACTED***"),
    (
        re.compile(
            r"((?:api[_-]?key|api[_-]?secret|token|password|signature)\s*[=:]\s*['\"]?)([^'\"\s,&]+)",
            re.IGNORECASE,
        ),
        r"\1***REDACTED***",
    ),
]


def register_secret(value: str | None) -> None:
    """Register a secret string so it is always redacted from logs."""
    if not value:
        return
    cleaned = value.strip()
    if len(cleaned) < 4:
        return
    with _SECRETS_LOCK:
        _REGISTERED_SECRETS.add(cleaned)


def register_env_secrets() -> None:
    """Scan current environment variables for sensitive keys and register their values."""
    for key, val in os.environ.items():
        upper = key.upper()
        if any(frag in upper for frag in _SENSITIVE_ENV_FRAGMENTS):
            register_secret(val)


def redact_text(text: str) -> str:
    """Replace any registered secret or known credential pattern with ***REDACTED***."""
    result = text
    with _SECRETS_LOCK:
        secrets = sorted(_REGISTERED_SECRETS, key=len, reverse=True)
    for secret in secrets:
        if secret in result:
            result = result.replace(secret, "***REDACTED***")
    for pattern, repl in _PATTERN_REDACTIONS:
        result = pattern.sub(repl, result)
    return result


class SecretRedactingFilter(logging.Filter):
    """Logging filter that scrubs secrets from messages and formatting args."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered = record.getMessage()
            redacted = redact_text(rendered)
            record.msg = redacted
            record.args = ()
            if record.exc_text:
                record.exc_text = redact_text(record.exc_text)
        except Exception:
            pass
        return True


class JsonFormatter(logging.Formatter):
    """Compact UTC JSON log formatter for container/production environments."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": redact_text(record.getMessage()),
        }
        if record.exc_info:
            payload["exc"] = redact_text(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure root logging for Universal Trader with secret redaction enabled."""
    register_env_secrets()
    root = logging.getLogger("universal_trader")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(SecretRedactingFilter())
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%SZ",
            )
        )
    root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the universal_trader namespace."""
    if not name.startswith("universal_trader"):
        name = f"universal_trader.{name}"
    logger = logging.getLogger(name)
    if not logging.getLogger("universal_trader").handlers:
        configure_logging()
    return logger
