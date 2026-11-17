"""
Authentication and authorization for Expert Decisions.
Enforces per-expert API keys mapped to expert names, stored as SHA-256 hashes,
and compared using constant-time hmac.compare_digest.
No header bypass (such as X-User-Role) is permitted.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import NamedTuple

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ExpertIdentity(NamedTuple):
    expert_id: str
    name: str
    role: str


# ---------------------------------------------------------------------------
# Expert credentials are supplied only through environment configuration.
# Never ship production expert keys in source control.
# EXPERT_API_KEYS_JSON format:
# {"raw-key": {"expert_id": "...", "name": "...", "role": "..."}}
# ---------------------------------------------------------------------------


def hash_key(raw_key: str) -> str:
    """Compute hex SHA-256 digest of raw key."""
    return hashlib.sha256(raw_key.strip().encode("utf-8")).hexdigest()


def verify_expert_api_key(provided_key: str | None) -> tuple[bool, ExpertIdentity | None]:
    """Verify an expert key against environment-configured credentials only."""
    import json

    if not provided_key or not provided_key.strip():
        return False, None

    settings = get_settings()
    registry: dict[str, ExpertIdentity] = {}

    try:
        configured = json.loads(settings.expert_api_keys_json or "{}")
        if not isinstance(configured, dict):
            raise ValueError("EXPERT_API_KEYS_JSON must be a JSON object")
        for raw_key, identity_data in configured.items():
            if not isinstance(identity_data, dict) or not raw_key:
                continue
            registry[hash_key(raw_key)] = ExpertIdentity(
                expert_id=str(identity_data.get("expert_id", "EXP-CONFIGURED")),
                name=str(identity_data.get("name", "Configured Regulatory Expert")),
                role=str(identity_data.get("role", "Designated Compliance Officer")),
            )
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("expert_registry_configuration_invalid", error=str(exc))
        return False, None

    # Optional single-key deployment mode. This key is never hardcoded in source.
    configured_key = (settings.expert_api_key or "").strip()
    if configured_key:
        registry[hash_key(configured_key)] = ExpertIdentity(
            expert_id="EXP-CONFIGURED-00",
            name="Configured Regulatory Expert",
            role="Designated Compliance Officer",
        )

    provided_hash = hash_key(provided_key)
    for stored_hash, identity in registry.items():
        if hmac.compare_digest(provided_hash, stored_hash):
            logger.info("expert_authenticated", expert_id=identity.expert_id, expert_name=identity.name)
            return True, identity

    logger.warning("expert_auth_failed", reason="invalid_key_or_unknown_hash")
    return False, None
