"""
Application configuration using Pydantic Settings.
All secrets come from environment variables — never hardcoded.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
REGULATIONS_DIR = DATA_DIR / "regulations"
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Central configuration — populated from env vars / .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "FSSAI Label Compliance Check Service"
    app_env: str = "development"
    app_debug: bool = False
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_log_level: str = "INFO"
    app_secret_key: str = ""
    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"

    # --- Database ---
    database_url: str = f"sqlite+aiosqlite:///{DATA_DIR / 'compliance.db'}"

    # --- LLM Provider ---
    llm_provider: str = "gemini"  # gemini | openai | mock (mock is tests/demo only)
    gemini_api_key: str = ""
    openai_api_key: str = ""
    openai_api_base: str = ""
    openai_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-3.8-flash"

    # --- Vision Extraction Provider ---
    vision_provider: str = "gemini"  # gemini | openai | mock (mock is for tests/demo only)
    gemini_vision_model: str = "gemini-3.8-flash"
    openai_vision_model: str = "gpt-4o-mini"
    extraction_confidence_threshold: float = 0.8
    upload_max_images: int = 4
    upload_max_size_mb: int = 5
    upload_retention_days: int = 7

    # --- Expert Decision Auth ---
    expert_api_key: str = ""
    expert_api_keys_json: str = "{}"

    # --- LLM Resilience ---
    llm_mock_fallback: bool = False  # never present a mocked answer as a live AI result
    llm_timeout_seconds: int = 30
    llm_max_retries: int = 3
    llm_circuit_breaker_threshold: int = 5
    llm_circuit_breaker_reset_seconds: int = 60

    # --- RAG / Vector Store ---
    chroma_persist_dir: str = str(DATA_DIR / "vectorstore")
    embedding_provider: str = "sentence-transformers"  # sentence-transformers | hash | mock
    embedding_model: str = "all-MiniLM-L6-v2"
    rag_top_k: int = 5
    rag_min_score: float = 0.3
    regulatory_require_authoritative_corpus: bool = True
    # JSON mapping of official PDF filename to independently verified SHA-256 digest.
    regulatory_authorized_sha256_json: str = "{}"

    # --- Rate Limiting ---
    rate_limit_requests: int = 30
    rate_limit_window_seconds: int = 60
    max_request_size_mb: int = 10

    # --- Metrics ---
    enable_prometheus: bool = True

    @field_validator("app_log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in allowed:
            raise ValueError(f"app_log_level must be one of {allowed}")
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


# ---------------------------------------------------------------------------
# Decision policy (loaded from YAML)
# ---------------------------------------------------------------------------

def load_decision_policy() -> dict[str, Any]:
    """Load the deterministic decision policy from YAML."""
    policy_path = CONFIG_DIR / "decision_policy.yaml"
    if not policy_path.exists():
        raise FileNotFoundError(f"Decision policy not found: {policy_path}")
    with open(policy_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_decision_policy() -> dict[str, Any]:
    """Cached accessor for the decision policy."""
    return load_decision_policy()


def get_disclaimer() -> str:
    """Return the disclaimer text from the decision policy."""
    policy = get_decision_policy()
    return policy.get("disclaimer", (
        "DISCLAIMER: This is an automated compliance pre-screening tool. "
        "It provides recommendations only and does NOT constitute regulatory approval. "
        "A qualified human regulatory expert must review all findings and make the "
        "final decision before any label is approved for printing."
    ))


# ---------------------------------------------------------------------------
# Singleton settings accessor
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()
