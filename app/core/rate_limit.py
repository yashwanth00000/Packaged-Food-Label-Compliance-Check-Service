"""Process-local sliding-window rate limiter used by the HTTP middleware."""
from __future__ import annotations

import time

_rate_limit_store: dict[str, list[float]] = {}

def check_rate_limit(client_ip: str, max_requests: int, window: int) -> bool:
    """Return True when the request is allowed, otherwise False."""
    now = time.time()
    recent = [stamp for stamp in _rate_limit_store.get(client_ip, []) if now - stamp < window]
    if len(recent) >= max_requests:
        _rate_limit_store[client_ip] = recent
        return False
    recent.append(now)
    _rate_limit_store[client_ip] = recent
    return True
