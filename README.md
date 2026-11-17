# FSSAI Food Label Compliance Check Service

> **DISCLAIMER:** This is an automated compliance pre-screening tool. It provides recommendations only and does **NOT** constitute regulatory approval. A qualified human regulatory expert must review all findings and make the final decision before any label is approved for printing.

## Overview

A web service that checks packaged food label details against FSSAI regulations, detects missing mandatory declarations and unsupported claims, and uses multiple AI agents to verify compliance with evidence and regulatory citations.

It returns one of three recommendations:
1. **CLEAR_FOR_PRINT** — No issues found (recommendation only)
2. **REVISE_AND_RECHECK** — Fixable issues detected
3. **REGULATORY_REVIEW** — Serious concerns requiring expert review

### Key Design Principles
- **Deterministic-first**: Mandatory fields, allergen consistency, and claim thresholds checked with Python rules. LLM only for ambiguous judgments.
- **Retrieval-first pipeline**: RAG retrieval runs before domain agents so analysis is grounded in indexed regulation chunks.
- **Fail-safe always**: Any error → `REGULATORY_REVIEW`. Never `CLEAR_FOR_PRINT`.
- **Human-in-the-loop**: Every output includes a disclaimer. Expert decisions logged with audit trail.
- **Explicit mock mode**: Mock providers are available for tests and demonstrations only; live scanning never silently reports mock output as real AI analysis.

## Architecture

```
Intake → Retrieval (top-k per field/claim) → Mandatory / Claims / Additives
    (deterministic rules + LLM fallback) → Critic → Decision (fully deterministic)
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | FastAPI + Uvicorn |
| Database | SQLite via SQLAlchemy (async) |
| Vector Store | ChromaDB |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| LLM | Gemini / OpenAI / MockLLM |
| Templates | Jinja2 + HTMX |
| Metrics | Prometheus |
| Testing | pytest + pytest-cov |

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Configure a real vision + reasoning provider before scanning (Gemini example)
copy .env.example .env
# Set GEMINI_API_KEY in .env; keep VISION_PROVIDER=gemini and LLM_PROVIDER=gemini

# Ingest the regulation corpus using deterministic hash embeddings
python scripts/ingest_regulations.py --mock-embeddings

# Run the service
uvicorn app.main:app --reload

# Open browser
# http://127.0.0.1:8000       — Web UI
# http://127.0.0.1:8000/docs  — API documentation
```

### Docker

```bash
# Build and run
docker compose up --build -d

# Check health
curl http://localhost:8000/health

# View logs
docker compose logs -f
```

### Using Make

```bash
make install    # Install deps
make ingest     # Ingest regulations
make dev        # Run with auto-reload
make test       # Run tests
make test-cov   # Run tests with coverage
make eval       # Run eval harness (mock mode)
make docker-up  # Build and start Docker
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/check` | Run compliance check on a label |
| `GET` | `/api/v1/check/{id}` | Fetch a specific report |
| `GET` | `/api/v1/checks` | List recent checks (paginated) |
| `POST` | `/api/v1/check/{id}/decision` | Submit expert decision |
| `GET` | `/api/v1/regulations/search?q=` | Search regulation chunks |
| `GET` | `/api/v1/samples` | List sample labels |
| `GET` | `/health` | Health check |
| `GET` | `/ready` | Readiness check |
| `GET` | `/metrics` | Prometheus metrics |

## Web UI Pages

| Path | Description |
|------|-------------|
| `/` | Home — submit a label for checking |
| `/check/{id}` | View result with expert review panel |
| `/history` | List of past checks |

## Sample Labels

Three sample labels are included for testing:

1. **Compliant Biscuit** — exercises the positive-path policy in mock tests. The live service will not issue `CLEAR_FOR_PRINT` until an authoritative, current FSSAI corpus has been ingested and verified.
2. **Missing Declarations** — missing allergens, net quantity, manufacturer, etc. → `REVISE_AND_RECHECK`
3. **Prohibited Claims** — "Cures cold and flu", "Clinically proven" → `REGULATORY_REVIEW`

## Testing

```bash
# Run all tests
pytest tests/ -q

# With coverage
pytest tests/ --cov=app --cov-report=term-missing

# Run specific test file
pytest tests/test_security.py -v
```

## Evaluation Harnesses

Dual evaluation harnesses supporting both deterministic mock mode (CI) and real Gemini LLM/Vision mode:

### 1. Label Compliance Evaluation (22 Samples)
```bash
# Mock mode (CI — no API key needed)
python -m eval.run_eval --mode mock

# Real LLM mode
GEMINI_API_KEY=your_key python -m eval.run_eval --mode real
```
Reports Recommendation Accuracy, Finding-Level Recall ($|\text{Retrieved} \cap \text{Expected}| / |\text{Expected}|$), per-class precision/recall, and citation validity.

### 2. Stage 0 Vision Extraction Evaluation (10 Images)
```bash
# Mock mode (CI)
python -m eval.run_extraction_eval --mode mock

# Real Vision LLM mode
GEMINI_API_KEY=your_key python -m eval.run_extraction_eval --mode real
```
Evaluates against `eval/images/ground_truth.json` reporting Field-Level Accuracy, Hallucination Rate (null $\to$ value), and Missed Field Rate (value $\to$ null).

## Photo Scan & Review Workflow (Stage 0)

1. **Capture or Upload Label Photos**: Open the live camera (rear camera preferred) and capture up to four panels, or select existing images. Camera access requires HTTPS or localhost and explicit browser permission.
2. **Upload Label Photos**: Drag and drop 1 to 4 clear photos (front, back, nutrition panel, ingredients panel). Supports JPEG, PNG, and WebP (up to 5 MB per image).
3. **Sanitization & Quality Gates**: Magic bytes content detection, EXIF/GPS metadata stripping, Lanczos downscaling (>2048px), and quality checks for extreme blur or darkness.
4. **Stage 0 Extraction**: Gemini Vision (or OpenAI / MockExtractor) extracts all 17 FSSAI fields following strict **Null-Not-Guess** policy (unseen fields return `null` with `confidence: 0.0`).
5. **Review & Confirm UI**: Pre-filled editable form shown side-by-side with zoomable image lightbox. Low confidence fields (< 0.80) highlighted in amber; missing fields badged in red. User must verify details against physical label before running compliance check. Only sanitized, EXIF-stripped image bytes are retained; uploaded label photos are not exposed as public static content.
6. **Decision Policy**: `CLEAR_FOR_PRINT` requires user confirmation, affirmative PASS on all mandatory checks, and zero unconfirmed low-confidence mandatory fields.
7. **Expert Actions & Audit Immutability**: Final decisions protected by `X-Expert-Key` header with SQLite database triggers and ORM listeners preventing any mutation or deletion of audit logs.

## Privacy & Image Retention Policy

- **EXIF & GPS Stripping**: All location and camera metadata are stripped from uploaded photos during intake.
- **Data Retention**: Label photos saved to `data/uploads/{extraction_id}/` are automatically scheduled for deletion after `UPLOAD_RETENTION_DAYS` (default: 7 days).
- **Security**: Label text from photos is treated as untrusted data (`<UNTRUSTED_DATA>`) to prevent prompt-injection attacks.

## Runtime mode and model configuration

The application defaults to Gemini for both camera-image extraction and compliance reasoning. Set `GEMINI_API_KEY` in `.env`; alternatively set `LLM_PROVIDER=openai` and `VISION_PROVIDER=openai` with `OPENAI_API_KEY` where supported. `VISION_PROVIDER=mock` and `LLM_PROVIDER=mock` are for deterministic tests and demonstrations only. Live provider failures do not silently fall back to mock answers (`LLM_MOCK_FALLBACK=false`). The camera scanner requires browser camera permission and HTTPS (or localhost); image upload remains available as a fallback.

The included regulation text files are sample extracts, not a complete authoritative FSSAI corpus. Do not use the result as legal approval until the corpus has been replaced with verified, versioned official regulations and reviewed by a qualified regulatory professional.

## Limitations

- **Vision Models**: Vision LLMs can misread small font sizes (e.g. 1mm text), curved surfaces, or reflective packaging glare. A qualified human must always cross-check extracted text against the physical packaging.
- **Sample Regulations**: The sample regulation text files in `data/regulations/` are marked `[SAMPLE]` and serve as representative demonstration extracts. Real FSSAI Gazette PDFs can be dropped into `data/regulations/` using the PDF ingestion utility.
- **Pre-screening Only**: This tool provides recommendations and does **NOT** constitute regulatory approval or legal advice. Final clearance must be granted by a certified food safety officer.

## Regulatory corpus status

The files currently shipped under `data/regulations/` are **sample/demo extracts**, not a complete or authoritative FSSAI corpus. See `data/regulations/REGULATORY_SOURCE_MANIFEST.md` for the official source index and amendment review notes. With the default production safety setting (`REGULATORY_REQUIRE_AUTHORITATIVE_CORPUS=true`), the decision engine refuses to issue `CLEAR_FOR_PRINT` when authoritative evidence is missing. Download, version, validate and ingest the applicable official FSSAI/Gazette instruments before using the service for a real label review. Future-effective amendments must not be applied before their commencement date. All outputs remain advisory and require qualified human review.

## Team Contributions

- **Yashwanth Gowda T G:** Project setup and deployment configuration, core runtime and data models, label extraction and compliance agents, and related tests.
- **Shreyan Ganesh:** API endpoints and web interface, including routes, templates, styling, and related API tests.
- **Krishna Koushik:** Regulatory data and ingestion, retrieval pipeline and services, and evaluation scripts and datasets.
