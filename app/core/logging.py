"""
Structured JSON logging with request/trace IDs and log redaction.
Uses structlog for structured logging with JSON output.
"""

from __future__ import annotations

import logging
import re
import sys
import uuid
from contextvars import ContextVar
from typing import Any

try:
    import structlog
except ImportError:  # minimal/local test environments may omit the optional formatter
    structlog = None  # type: ignore[assignment]


class _FallbackLogger:
    """Small stdlib adapter preserving structlog's event + keyword API."""
    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def bind(self, **kwargs):
        return self

    def _log(self, level: int, event: str, **kwargs):
        if kwargs:
            self._logger.log(level, "%s | %s", event, kwargs)
        else:
            self._logger.log(level, "%s", event)

    def debug(self, event: str, **kwargs): self._log(logging.DEBUG, event, **kwargs)
    def info(self, event: str, **kwargs): self._log(logging.INFO, event, **kwargs)
    def warning(self, event: str, **kwargs): self._log(logging.WARNING, event, **kwargs)
    def error(self, event: str, **kwargs): self._log(logging.ERROR, event, **kwargs)
    def exception(self, event: str, **kwargs): self._log(logging.ERROR, event, **kwargs)

# ---------------------------------------------------------------------------
# Context variables for request-scoped trace IDs
# ---------------------------------------------------------------------------
_request_id_var: ContextVar[str] = ContextVar("request_id", default="")
_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def get_request_id() -> str:
    return _request_id_var.get()


def set_request_id(request_id: str | None = None) -> str:
    rid = request_id or uuid.uuid4().hex[:12]
    _request_id_var.set(rid)
    return rid


def get_trace_id() -> str:
    return _trace_id_var.get()


def set_trace_id(trace_id: str | None = None) -> str:
    tid = trace_id or uuid.uuid4().hex[:16]
    _trace_id_var.set(tid)
    return tid


# ---------------------------------------------------------------------------
# Sensitive field redaction
# ---------------------------------------------------------------------------
_REDACT_PATTERNS = [
    re.compile(r"(api[_-]?key\s*[=:]\s*)\S+", re.IGNORECASE),
    re.compile(r"(secret\s*[=:]\s*)\S+", re.IGNORECASE),
    re.compile(r"(password\s*[=:]\s*)\S+", re.IGNORECASE),
    re.compile(r"(token\s*[=:]\s*)\S+", re.IGNORECASE),
]

_REDACT_KEYS = {"api_key", "apikey", "secret", "password", "token",
                "gemini_api_key", "openai_api_key", "app_secret_key"}


def _redact_sensitive(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Structlog processor that redacts sensitive values from log events."""
    for key in list(event_dict.keys()):
        if key.lower() in _REDACT_KEYS:
            event_dict[key] = "***REDACTED***"
        elif isinstance(event_dict[key], str):
            for pattern in _REDACT_PATTERNS:
                event_dict[key] = pattern.sub(r"\1***REDACTED***", event_dict[key])
    return event_dict


def _add_request_context(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Inject request_id and trace_id into every log event."""
    rid = get_request_id()
    tid = get_trace_id()
    if rid:
        event_dict["request_id"] = rid
    if tid:
        event_dict["trace_id"] = tid
    return event_dict


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging(log_level: str = "INFO", json_output: bool = True) -> None:
    """Configure structlog and stdlib logging for the application."""
    if sys.platform == "win32":
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    level = getattr(logging, log_level.upper(), logging.INFO)

    if structlog is None:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
        return

    # Shared processors
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_request_context,
        _redact_sensitive,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging to use structlog formatting
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # Reduce noise from third-party loggers
    for name in ("uvicorn.access", "chromadb", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str = __name__):
    """Get a structured logger, with stdlib fallback when structlog is unavailable."""
    return structlog.get_logger(name) if structlog is not None else _FallbackLogger(name)
