# Rebuild report — Packaged-Food-Label-Compliance-Check-Service

## Product workflow

1. Capture up to four package panels with a mobile/laptop browser camera, or select existing image files.
2. Validate images by file signature, reject unsupported/corrupt/oversized inputs, strip EXIF metadata, normalize/downscale images and evaluate basic visual quality.
3. Send sanitized images to the configured vision provider (Gemini by default) to extract structured label fields. Unreadable values must remain null rather than be guessed.
4. Show extracted fields for operator review and confirmation before compliance analysis.
5. Retrieve regulation chunks and run mandatory-declaration, claims, additive/allergen, critic and deterministic decision checks.
6. Produce an advisory report with findings, evidence/citations and a recommendation. A qualified regulatory expert remains responsible for final review.

## Rebuild changes

- Added an explicit live-camera UI with capture, preview, removal, multi-panel capture (up to four images), stop-camera control and file-upload fallback.
- Made Gemini the default vision and reasoning provider; migrated the integration to Google's `google-genai` SDK. Mock providers are explicit test/demo settings and are not an automatic production fallback.
- Added visible configuration failure/manual-entry handling when a real vision provider is unavailable.
- Persist only sanitized images; removed the public static `/uploads` mount and added a controlled image-serving endpoint keyed by high-entropy extraction IDs.
- Added upload size bounds, retention cleanup and empty extraction-folder cleanup.
- Removed hardcoded expert API keys; expert credentials now come from environment configuration.
- Restricted CORS to configured origins and set Docker production debug/log settings explicitly.
- Separated Docker database, vector index and uploaded-image volumes so they do not mask the bundled regulation corpus.
- Made regulatory evidence fail closed: sample `.txt` documents and unverified PDFs cannot support `CLEAR_FOR_PRINT`. A PDF only counts as authoritative when its SHA-256 matches an explicit deployment allowlist.
- Added a regulatory source manifest documenting official FSSAI source indexes and amendment review notes.
- Added tests for sample-only/missing-corpus decision gating and factored rate-limiting state into a lightweight module.

## Configuration before live scanning

1. Copy `.env.example` to `.env`.
2. Set `GEMINI_API_KEY` (or configure a supported OpenAI provider for both LLM and vision).
3. Keep `VISION_PROVIDER=gemini`, `LLM_PROVIDER=gemini`, `LLM_MOCK_FALLBACK=false` for real AI analysis.
4. For actual compliance work, obtain and validate the applicable official FSSAI/Gazette PDFs, reconcile amendments/effective dates, and have a qualified specialist review them. Put each reviewed PDF filename and SHA-256 digest in `REGULATORY_AUTHORIZED_SHA256_JSON` before ingestion. Do not add drafts or future-effective changes as current law.
5. Run `python scripts/ingest_regulations.py --mock-embeddings` locally, or `docker compose up --build` with the configured `.env`.
6. Open `http://127.0.0.1:8000`. Camera capture requires browser camera permission and a secure context (HTTPS or localhost).

## Validation performed in this rebuild environment

- Python bytecode compilation: passed (`python -m compileall -q app eval scripts tests`).
- Focused image-processing, regulatory-gate and retention tests: **12 passed**.
- Docker Compose YAML parse and volume/environment inspection: passed.
- Full pytest suite: **not completed**. Three test modules failed during collection because `chromadb` is not installed in this environment. Install `requirements.txt` before running the full suite.
- Browser camera capture and JavaScript execution: not exercised against a physical camera/browser in this environment.
- Docker build/startup and live Gemini calls: not executed in this environment. Supply credentials and test in the target environment.

## Regulatory and legal limitation

The included `.txt` regulatory documents remain sample/demo extracts. The manifest documents official FSSAI source indexes but does not mean a complete consolidated legal corpus is already downloaded, reconciled, reviewed or indexed. Until verified PDFs are added and allowlisted, the production decision engine intentionally blocks `CLEAR_FOR_PRINT` and returns `REGULATORY_REVIEW` when authoritative evidence is missing. All results are advisory, not FSSAI approval.
