"""
FastAPI application entry point.
Lifespan management, middleware, and route registration.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import PROJECT_ROOT, get_disclaimer, get_settings
from app.core.errors import register_error_handlers
from app.core.logging import get_logger, set_request_id, set_trace_id, setup_logging
from app.core.metrics import (
    APP_INFO,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    metrics_response,
)
from app.models.database import close_db, init_db

logger = get_logger(__name__)

from app.core.rate_limit import _rate_limit_store, check_rate_limit

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown logic."""
    settings = get_settings()
    setup_logging(
        log_level=settings.app_log_level,
        json_output=settings.is_production,
    )
    logger.info("app_starting", env=settings.app_env, provider=settings.llm_provider)

    APP_INFO.info({
        "version": "1.0.0",
        "env": settings.app_env,
        "llm_provider": settings.llm_provider,
    })

    # Initialise database
    await init_db()
    logger.info("database_initialised")

    # Clean expired private label photos at startup and once per day.
    from app.core.retention import cleanup_old_uploads
    await asyncio.to_thread(cleanup_old_uploads)

    async def retention_loop() -> None:
        while True:
            await asyncio.sleep(24 * 60 * 60)
            await asyncio.to_thread(cleanup_old_uploads)

    retention_task = asyncio.create_task(retention_loop(), name="label-photo-retention")

    # Docker compose ingests regulations after persistent volumes are mounted.
    # Local development can run scripts/ingest_regulations.py explicitly.
    logger.info("app_ready")

    try:
        yield
    finally:
        retention_task.cancel()
        try:
            await retention_task
        except asyncio.CancelledError:
            pass
        # Shutdown
        await close_db()
    logger.info("app_shutdown")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=(
            "Checks packaged food labels against FSSAI regulations. "
            "Provides recommendations only — a human regulatory expert "
            "makes the final decision."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # --- CORS ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Error handlers ---
    register_error_handlers(app)

    # --- Middleware: request ID, logging, metrics, rate limiting, size limit ---
    @app.middleware("http")
    async def request_middleware(request: Request, call_next) -> Response:
        # Request ID
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        set_request_id(request_id)
        set_trace_id()

        # Request size limit
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.max_request_size_mb * 1024 * 1024:
            return Response(
                status_code=413,
                content='{"error": "Request too large"}',
                media_type="application/json",
            )

        # Rate limiting
        client_ip = request.client.host if request.client else "unknown"
        if not check_rate_limit(client_ip, settings.rate_limit_requests, settings.rate_limit_window_seconds):
            return Response(
                status_code=429,
                content='{"error": "Rate limit exceeded. Please try again later."}',
                media_type="application/json",
            )

        # Timing
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start

        # Metrics
        endpoint = request.url.path
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(duration)

        # Response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{duration:.4f}"
        return response

    # --- Static files ---
    static_dir = PROJECT_ROOT / "app" / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Uploaded label photos are served through a controlled application route,
    # not exposed as a public static directory.

    # --- Register routers (deferred import to avoid circular deps) ---
    from app.api.routes import api_router
    from app.ui.routes import ui_router

    app.include_router(api_router)
    app.include_router(ui_router)

    # --- Health / ready / metrics ---
    @app.get("/health", tags=["ops"])
    async def health():
        return {
            "status": "ok",
            "version": "1.0.0",
            "disclaimer": get_disclaimer(),
        }

    @app.get("/ready", tags=["ops"])
    async def ready():
        return {
            "ready": True,
            "database": "ok",
            "vector_store": "ok",
            "llm_provider": settings.llm_provider,
        }

    @app.get("/metrics", tags=["ops"])
    async def metrics():
        return metrics_response()

    return app


# ---------------------------------------------------------------------------
# Module-level app instance for uvicorn
# ---------------------------------------------------------------------------
app = create_app()
