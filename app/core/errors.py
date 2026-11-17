"""
Custom exceptions and global FastAPI error handlers.
No stack traces exposed to users — structured error responses only.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom exception hierarchy
# ---------------------------------------------------------------------------

class AppError(Exception):
    """Base application error."""

    def __init__(
        self,
        message: str = "An internal error occurred",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: dict[str, Any] | None = None,
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(AppError):
    """Input validation error."""

    def __init__(self, message: str = "Validation error", details: dict[str, Any] | None = None):
        super().__init__(message=message, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, details=details)


class NotFoundError(AppError):
    """Resource not found."""

    def __init__(self, message: str = "Resource not found"):
        super().__init__(message=message, status_code=status.HTTP_404_NOT_FOUND)


class RateLimitError(AppError):
    """Rate limit exceeded."""

    def __init__(self, message: str = "Rate limit exceeded. Please try again later."):
        super().__init__(message=message, status_code=status.HTTP_429_TOO_MANY_REQUESTS)


class LLMError(AppError):
    """LLM provider error (timeout, auth, malformed response)."""

    def __init__(self, message: str = "LLM service error", details: dict[str, Any] | None = None):
        super().__init__(message=message, status_code=status.HTTP_502_BAD_GATEWAY, details=details)


class LLMTimeoutError(LLMError):
    """LLM call timed out."""

    def __init__(self, message: str = "LLM request timed out"):
        super().__init__(message=message)


class CircuitBreakerOpenError(LLMError):
    """Circuit breaker is open — falling back."""

    def __init__(self, message: str = "LLM circuit breaker open, using fallback"):
        super().__init__(message=message)


class RAGError(AppError):
    """RAG / vector store error."""

    def __init__(self, message: str = "Retrieval service error"):
        super().__init__(message=message, status_code=status.HTTP_502_BAD_GATEWAY)


class AgentError(AppError):
    """An agent in the pipeline failed."""

    def __init__(self, agent_name: str, message: str = "Agent processing failed"):
        super().__init__(
            message=f"Agent '{agent_name}' failed: {message}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"agent": agent_name},
        )


class PipelineError(AppError):
    """The orchestration pipeline failed."""

    def __init__(self, message: str = "Compliance check pipeline failed"):
        super().__init__(message=message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ---------------------------------------------------------------------------
# Global exception handlers — register on the FastAPI app
# ---------------------------------------------------------------------------

def register_error_handlers(app: FastAPI) -> None:
    """Attach global exception handlers to the FastAPI application."""

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        logger.warning(
            "app_error",
            error_type=type(exc).__name__,
            message=exc.message,
            status_code=exc.status_code,
            path=str(request.url),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": type(exc).__name__,
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        # Log full traceback internally but never expose to user
        logger.error(
            "unhandled_exception",
            error_type=type(exc).__name__,
            path=str(request.url),
            exc_info=exc,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred. Please try again later.",
                "details": {},
            },
        )
