"""
LLM provider interface with Gemini, OpenAI-compatible, and MockLLM implementations.
Includes retry with backoff, timeout, and circuit breaker.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any

from app.core.config import get_settings
from app.core.errors import LLMError
from app.core.logging import get_logger
from app.core.metrics import (
    LLM_CALLS,
    LLM_CIRCUIT_BREAKER_STATE,
    LLM_LATENCY,
    LLM_TOKEN_USAGE,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Simple circuit breaker for LLM calls."""

    def __init__(self, threshold: int = 5, reset_seconds: int = 60):
        self.threshold = threshold
        self.reset_seconds = reset_seconds
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.is_open = False

    def record_success(self):
        self.failure_count = 0
        self.is_open = False

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.threshold:
            self.is_open = True

    def can_proceed(self) -> bool:
        if not self.is_open:
            return True
        # Check if enough time has passed to try again
        if time.time() - self.last_failure_time > self.reset_seconds:
            self.is_open = False
            self.failure_count = 0
            return True
        return False


# ---------------------------------------------------------------------------
# Abstract LLM Provider
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Abstract LLM provider interface."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """
        Generate a response.
        Returns: {"content": str, "tokens_used": int, "provider": str}
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...


# ---------------------------------------------------------------------------
# Mock LLM — deterministic, rule-based responses for offline testing
# ---------------------------------------------------------------------------

class MockLLM(LLMProvider):
    """
    Deterministic mock LLM that produces structured responses based on
    keyword matching. Works offline with no API key. First-class citizen.
    """

    @property
    def provider_name(self) -> str:
        return "mock"

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        prompt_lower = prompt.lower()

        # Determine which agent is calling based on system prompt keywords
        if "intake" in system_prompt.lower() or "normalize" in system_prompt.lower():
            content = self._mock_intake(prompt_lower, prompt)
        elif "critic" in system_prompt.lower() or "verif" in system_prompt.lower():
            content = self._mock_critic(prompt_lower, prompt)
        elif "claim" in system_prompt.lower():
            content = self._mock_claims(prompt_lower, prompt)
        elif "mandatory" in system_prompt.lower():
            content = self._mock_mandatory(prompt_lower, prompt)
        elif "additive" in system_prompt.lower() or "allergen" in system_prompt.lower():
            content = self._mock_additives(prompt_lower, prompt)
        else:
            content = json.dumps({"response": "Mock LLM response", "confidence": 0.5})

        return {
            "content": content,
            "tokens_used": len(prompt.split()) + len(content.split()),
            "provider": "mock",
        }

    def _mock_intake(self, prompt_lower: str, prompt: str) -> str:
        """Mock intake agent response."""
        issues = []
        if "ignore previous" in prompt_lower or "ignore all" in prompt_lower:
            issues.append("Potential prompt injection detected in label text — flagged for review")

        return json.dumps({
            "normalized": True,
            "issues": issues,
            "confidence": 0.95,
        })

    def _mock_claims(self, prompt_lower: str, prompt: str) -> str:
        """Mock claims verification — produces structured claim evaluation."""
        claim_match = re.search(r'---\n(.*?)\n---', prompt, re.DOTALL)
        claim_text = claim_match.group(1).lower().strip() if claim_match else prompt_lower

        prohibited_patterns = [r"\bcure[s]?\b", r"\bprevent[s]?\b", r"\btreat[s]?\b", r"100%\s*natural", r"\bmiracle\b"]
        if any(re.search(p, claim_text) for p in prohibited_patterns):
            return json.dumps({
                "status": "PROHIBITED",
                "reason": "Medicinal or absolute claims are prohibited under FSSAI Advertising Regulations.",
                "confidence": 0.95,
            })
        if "unsupported" in claim_text or "fake" in claim_text:
            return json.dumps({
                "status": "UNSUPPORTED",
                "reason": "Claim is not substantiated by nutritional values or ingredient composition.",
                "confidence": 0.85,
            })
        if "ambiguous" in claim_text or "unclear" in claim_text:
            return json.dumps({
                "status": "NEEDS_EVIDENCE",
                "reason": "Claim requires further expert review due to ambiguous wording",
                "confidence": 0.6,
            })
        return json.dumps({
            "status": "SUPPORTED",
            "reason": "Claim appears consistent with regulation conditions and ingredient profile.",
            "confidence": 0.9,
        })

    def _mock_mandatory(self, prompt_lower: str, prompt: str) -> str:
        """Mock mandatory declaration check."""
        return json.dumps({
            "findings": [],
            "confidence": 0.9,
        })

    def _mock_additives(self, prompt_lower: str, prompt: str) -> str:
        """Mock additives/allergen check."""
        return json.dumps({
            "findings": [],
            "confidence": 0.85,
        })

    def _mock_critic(self, prompt_lower: str, prompt: str) -> str:
        """
        Mock critic agent. Reviews findings and may discard unsupported ones.
        Looks for findings without evidence or with low confidence.
        """
        try:
            # Try to extract findings from the prompt
            findings_match = re.search(r'\[.*\]', prompt, re.DOTALL)
            if findings_match:
                findings = json.loads(findings_match.group())
                verified = []
                removed = []
                for f in findings:
                    # Remove findings without evidence
                    if not f.get("evidence_quote") and not f.get("regulation_name"):
                        removed.append(f.get("finding_id", "unknown"))
                        continue
                    # Remove very low confidence findings
                    if f.get("confidence", 0.5) < 0.2:
                        removed.append(f.get("finding_id", "unknown"))
                        continue
                    f["verified_by_critic"] = True
                    verified.append(f)
                return json.dumps({
                    "verified_findings": verified,
                    "removed_finding_ids": removed,
                    "confidence": 0.85,
                })
        except (json.JSONDecodeError, AttributeError):
            pass

        return json.dumps({
            "verified_findings": [],
            "removed_finding_ids": [],
            "confidence": 0.5,
        })


# ---------------------------------------------------------------------------
# Gemini LLM Provider
# ---------------------------------------------------------------------------

class GeminiLLM(LLMProvider):
    """Google Gemini API provider."""

    def __init__(self):
        settings = get_settings()
        if not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        try:
            from google import genai
            from google.genai import types
            self._client = genai.Client(api_key=settings.gemini_api_key)
            self._types = types
            self._model = settings.gemini_model
        except ImportError as exc:
            raise LLMError("google-genai package is required for Gemini provider") from exc

    @property
    def provider_name(self) -> str:
        return "gemini"

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=full_prompt,
                config=self._types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            content = response.text
            tokens = getattr(response, 'usage_metadata', None)
            total_tokens = 0
            if tokens:
                total_tokens = getattr(tokens, 'total_token_count', 0)
            return {
                "content": content,
                "tokens_used": total_tokens,
                "provider": "gemini",
            }
        except Exception as e:
            raise LLMError(f"Gemini API error: {e!s}")


# ---------------------------------------------------------------------------
# OpenAI-Compatible LLM Provider
# ---------------------------------------------------------------------------

class OpenAILLM(LLMProvider):
    """OpenAI-compatible API provider."""

    def __init__(self):
        settings = get_settings()
        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        try:
            from openai import OpenAI
            kwargs = {"api_key": settings.openai_api_key}
            if settings.openai_api_base:
                kwargs["base_url"] = settings.openai_api_base
            self._client = OpenAI(**kwargs)
            self._model = settings.openai_model
        except ImportError:
            raise LLMError("openai package is required for OpenAI provider")

    @property
    def provider_name(self) -> str:
        return "openai"

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0
            return {
                "content": content,
                "tokens_used": tokens,
                "provider": "openai",
            }
        except Exception as e:
            raise LLMError(f"OpenAI API error: {e!s}")


# ---------------------------------------------------------------------------
# LLM Service (with retry, circuit breaker, fallback)
# ---------------------------------------------------------------------------

class LLMService:
    """
    LLM service with retries and circuit breaker. Mock fallback is opt-in only.
    """

    def __init__(self, provider: LLMProvider | None = None):
        settings = get_settings()
        self._circuit_breaker = CircuitBreaker(
            threshold=settings.llm_circuit_breaker_threshold,
            reset_seconds=settings.llm_circuit_breaker_reset_seconds,
        )
        self._fallback = MockLLM()

        if provider is not None:
            self._provider = provider
        elif settings.llm_provider == "gemini":
            self._provider = GeminiLLM()
        elif settings.llm_provider == "openai":
            self._provider = OpenAILLM()
        else:
            self._provider = MockLLM()

        self._max_retries = settings.llm_max_retries
        self._timeout = settings.llm_timeout_seconds

    @property
    def provider_name(self) -> str:
        return self._provider.provider_name

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """
        Generate with retry, circuit breaker, and fallback.
        When a live provider fails, raises unless LLM_MOCK_FALLBACK is explicitly enabled.
        """
        # Check circuit breaker
        if not self._circuit_breaker.can_proceed():
            LLM_CIRCUIT_BREAKER_STATE.labels(provider=self._provider.provider_name).set(1)
            logger.warning("circuit_breaker_open", provider=self._provider.provider_name)
            if get_settings().llm_mock_fallback:
                return self._fallback_generate(prompt, system_prompt, temperature, max_tokens)
            raise LLMError(f"LLM circuit breaker is open for {self._provider.provider_name}; refusing mock fallback")

        LLM_CIRCUIT_BREAKER_STATE.labels(provider=self._provider.provider_name).set(0)

        start = time.time()
        try:
            result = self._call_with_retry(prompt, system_prompt, temperature, max_tokens)
            duration = time.time() - start

            self._circuit_breaker.record_success()
            LLM_CALLS.labels(provider=result["provider"], status="success").inc()
            LLM_LATENCY.labels(provider=result["provider"]).observe(duration)
            LLM_TOKEN_USAGE.labels(provider=result["provider"], direction="total").inc(
                result.get("tokens_used", 0)
            )
            return result

        except Exception as e:
            duration = time.time() - start
            self._circuit_breaker.record_failure()
            LLM_CALLS.labels(provider=self._provider.provider_name, status="error").inc()
            LLM_LATENCY.labels(provider=self._provider.provider_name).observe(duration)
            logger.error(
                "llm_call_failed",
                provider=self._provider.provider_name,
                error=str(e),
                duration=duration,
            )
            if get_settings().llm_mock_fallback:
                return self._fallback_generate(prompt, system_prompt, temperature, max_tokens)
            raise LLMError(f"LLM provider {self._provider.provider_name} failed; mock fallback is disabled: {e!s}") from e

    def _call_with_retry(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Retry wrapper around the provider's generate method."""
        last_error = None
        for attempt in range(self._max_retries):
            try:
                return self._provider.generate(prompt, system_prompt, temperature, max_tokens)
            except Exception as e:
                last_error = e
                logger.warning(
                    "llm_retry",
                    attempt=attempt + 1,
                    max_retries=self._max_retries,
                    error=str(e),
                )
        raise last_error or LLMError("LLM call failed after retries")

    def _fallback_generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Fall back to MockLLM when primary provider fails."""
        logger.warning("llm_fallback_to_mock", primary=self._provider.provider_name)
        result = self._fallback.generate(prompt, system_prompt, temperature, max_tokens)
        result["fallback"] = True
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_llm_service: LLMService | None = None


def get_llm_service(provider: LLMProvider | None = None) -> LLMService:
    """Get or create the singleton LLM service."""
    global _llm_service
    if _llm_service is None or provider is not None:
        _llm_service = LLMService(provider=provider)
    return _llm_service


def reset_llm_service() -> None:
    """Reset the LLM service (for testing)."""
    global _llm_service
    _llm_service = None
