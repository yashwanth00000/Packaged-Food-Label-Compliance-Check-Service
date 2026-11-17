"""
Prometheus metrics for the FSSAI compliance check service.
Tracks request counts, latencies, agent performance, and recommendation distribution.
"""

from __future__ import annotations

from fastapi import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, Info, generate_latest

# ---------------------------------------------------------------------------
# Application-level metrics
# ---------------------------------------------------------------------------

APP_INFO = Info("fssai_app", "Application metadata")

REQUEST_COUNT = Counter(
    "fssai_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "fssai_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

# ---------------------------------------------------------------------------
# Agent-level metrics
# ---------------------------------------------------------------------------

AGENT_LATENCY = Histogram(
    "fssai_agent_duration_seconds",
    "Per-agent processing latency",
    ["agent_name"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

AGENT_ERRORS = Counter(
    "fssai_agent_errors_total",
    "Total agent errors",
    ["agent_name", "error_type"],
)

AGENT_RETRIES = Counter(
    "fssai_agent_retries_total",
    "Total agent retry attempts",
    ["agent_name"],
)

# ---------------------------------------------------------------------------
# LLM metrics
# ---------------------------------------------------------------------------

LLM_CALLS = Counter(
    "fssai_llm_calls_total",
    "Total LLM API calls",
    ["provider", "status"],
)

LLM_LATENCY = Histogram(
    "fssai_llm_call_duration_seconds",
    "LLM call latency",
    ["provider"],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)

LLM_TOKEN_USAGE = Counter(
    "fssai_llm_tokens_total",
    "Total LLM tokens used",
    ["provider", "direction"],  # direction: input | output
)

LLM_CIRCUIT_BREAKER_STATE = Gauge(
    "fssai_llm_circuit_breaker_open",
    "Whether the LLM circuit breaker is open (1) or closed (0)",
    ["provider"],
)

# ---------------------------------------------------------------------------
# Compliance check metrics
# ---------------------------------------------------------------------------

CHECK_COUNT = Counter(
    "fssai_compliance_checks_total",
    "Total compliance checks processed",
    ["status"],  # success | error
)

RECOMMENDATION_DISTRIBUTION = Counter(
    "fssai_recommendations_total",
    "Distribution of compliance recommendations",
    ["recommendation"],
)

FINDING_COUNT = Counter(
    "fssai_findings_total",
    "Total findings by severity",
    ["severity", "issue_type"],
)

CRITIC_REMOVALS = Counter(
    "fssai_critic_removals_total",
    "Findings removed by the critic agent",
)

# ---------------------------------------------------------------------------
# RAG metrics
# ---------------------------------------------------------------------------

RAG_QUERIES = Counter(
    "fssai_rag_queries_total",
    "Total RAG retrieval queries",
    ["status"],
)

RAG_LATENCY = Histogram(
    "fssai_rag_query_duration_seconds",
    "RAG query latency",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)

# ---------------------------------------------------------------------------
# Active checks
# ---------------------------------------------------------------------------

ACTIVE_CHECKS = Gauge(
    "fssai_active_checks",
    "Number of currently running compliance checks",
)

# ---------------------------------------------------------------------------
# Vision Extraction metrics
# ---------------------------------------------------------------------------

EXTRACTION_REQUESTS = Counter(
    "fssai_extraction_requests_total",
    "Total photo extraction requests",
    ["status"],
)

EXTRACTION_LATENCY = Histogram(
    "fssai_extraction_duration_seconds",
    "Label photo extraction latency in seconds",
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0),
)

EXTRACTION_CONFIDENCE = Histogram(
    "fssai_extraction_field_confidence",
    "Distribution of vision extraction confidence scores",
    buckets=(0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0),
)


# ---------------------------------------------------------------------------
# Metrics endpoint helper
# ---------------------------------------------------------------------------

def metrics_response() -> Response:
    """Generate a Prometheus-format metrics response."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
