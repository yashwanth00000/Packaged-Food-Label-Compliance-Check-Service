"""
RAG ingestion: chunk regulation documents, embed, and index into ChromaDB.
Metadata per chunk: regulation_name, clause, source_file, page, version.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import REGULATIONS_DIR, get_settings
from app.core.logging import get_logger
from app.rag.embeddings import EmbeddingProvider, get_embedding_provider

logger = get_logger(__name__)

COLLECTION_NAME = "fssai_regulations"


def _chunk_document(
    text: str,
    source_file: str,
    page_number: int = 1,
    is_sample: bool = True,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[dict[str, Any]]:
    """
    Chunk a regulation document into indexed segments with metadata.
    Uses clause boundaries when possible, falls back to character-based chunking.
    Preserves page_number and is_sample metadata.
    """
    chunks: list[dict[str, Any]] = []

    # Try to extract regulation name from the document
    reg_name = "Unknown Regulation"
    reg_match = re.search(r"REGULATION:\s*(.+)", text)
    if reg_match:
        reg_name = reg_match.group(1).strip()
    elif is_sample:
        reg_name = f"[SAMPLE] {source_file.replace('.txt', '').replace('_', ' ').title()}"

    version = ""
    ver_match = re.search(r"VERSION:\s*(.+)", text)
    if ver_match:
        version = ver_match.group(1).strip()

    # Split by CLAUSE markers for semantic chunking
    clause_pattern = re.compile(r"(CLAUSE\s+\d+[A-Za-z]?:\s*[^\n]+)")
    parts = clause_pattern.split(text)

    current_clause = ""
    clause_texts: list[tuple[str, str]] = []  # (clause_header, text)

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if clause_pattern.match(part):
            current_clause = part
        else:
            if current_clause:
                clause_texts.append((current_clause, part))
                current_clause = ""
            else:
                # Preamble or non-clause text
                clause_texts.append(("", part))

    # If no clauses found, do character-based chunking
    if not clause_texts:
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i : i + chunk_size].strip()
            if len(chunk_text) < 20:
                continue
            chunk_id = hashlib.md5(
                f"{source_file}:p{page_number}:{i}".encode()
            ).hexdigest()[:16]
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "regulation_name": reg_name,
                    "clause": "",
                    "source_file": source_file,
                    "page_number": page_number,
                    "is_sample": is_sample,
                    "version": version,
                },
            })
        return chunks

    # Process clause-based chunks
    for clause_header, clause_body in clause_texts:
        full_text = f"{clause_header}\n{clause_body}".strip() if clause_header else clause_body.strip()
        if len(full_text) < 20:
            continue

        # Extract clause number
        clause_num = ""
        if clause_header:
            cm = re.match(r"CLAUSE\s+(\d+[A-Za-z]?)", clause_header)
            if cm:
                clause_num = f"Clause {cm.group(1)}"

        # If clause text is too long, sub-chunk it
        if len(full_text) > chunk_size * 2:
            for i in range(0, len(full_text), chunk_size - overlap):
                sub = full_text[i : i + chunk_size].strip()
                if len(sub) < 20:
                    continue
                chunk_id = hashlib.md5(
                    f"{source_file}:p{page_number}:{clause_num}:{i}".encode()
                ).hexdigest()[:16]
                chunks.append({
                    "id": chunk_id,
                    "text": sub,
                    "metadata": {
                        "regulation_name": reg_name,
                        "clause": clause_num,
                        "source_file": source_file,
                        "page_number": page_number,
                        "is_sample": is_sample,
                        "version": version,
                    },
                })
        else:
            chunk_id = hashlib.md5(
                f"{source_file}:p{page_number}:{clause_num}".encode()
            ).hexdigest()[:16]
            chunks.append({
                "id": chunk_id,
                "text": full_text,
                "metadata": {
                    "regulation_name": reg_name,
                    "clause": clause_num,
                    "source_file": source_file,
                    "page_number": page_number,
                    "is_sample": is_sample,
                    "version": version,
                },
            })

    return chunks


def _extract_text_from_pdf(filepath: Path) -> tuple[list[tuple[int, str]], list[str]]:
    """
    Extract (page_num, text) from a PDF file using pypdf.
    Detects scanned pages with insufficient extracted text and issues warnings.
    Returns (pages, scanned_warnings).
    """
    pages: list[tuple[int, str]] = []
    scanned_warnings: list[str] = []
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(filepath))
        for idx, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            word_count = len(text.split())
            if word_count < 10:
                warning_msg = (
                    f"Page {idx + 1} of '{filepath.name}' contains only {word_count} extractable words. "
                    "It appears to be a scanned image or non-OCR page. Text cannot be indexed without OCR."
                )
                scanned_warnings.append(warning_msg)
                logger.warning(
                    "scanned_pdf_page_detected",
                    file=filepath.name,
                    page=idx + 1,
                    words=word_count,
                )
            if text:
                pages.append((idx + 1, text))
    except Exception as e:
        logger.error("pdf_extraction_error", file=filepath.name, error=str(e))
    return pages, scanned_warnings


def get_chroma_client(persist_dir: str | None = None) -> chromadb.ClientAPI:
    """Get or create a persistent ChromaDB client."""
    if persist_dir is None:
        persist_dir = get_settings().chroma_persist_dir
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def ingest_regulations(
    regulations_dir: str | Path | None = None,
    persist_dir: str | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    use_mock_embeddings: bool = False,
) -> dict[str, Any]:
    """
    Scan the regulations directory for .txt and .pdf files, chunk each, embed, and index.
    Supports real FSSAI PDFs with per-page chunking and page numbers.
    Returns summary statistics.
    """
    if regulations_dir is None:
        regulations_dir = REGULATIONS_DIR
    regulations_dir = Path(regulations_dir)

    if embedding_provider is None:
        embedding_provider = get_embedding_provider(use_mock=use_mock_embeddings)

    client = get_chroma_client(persist_dir)

    # Delete existing collection if it exists (full re-index)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks: list[dict[str, Any]] = []
    files_processed = 0

    # Ingest text files (always sample-only)
    for filepath in sorted(regulations_dir.glob("*.txt")):
        logger.info("ingesting_text_file", file=filepath.name)
        text = filepath.read_text(encoding="utf-8")
        file_chunks = _chunk_document(
            text,
            source_file=filepath.name,
            page_number=1,
            is_sample=True,
        )
        all_chunks.extend(file_chunks)
        files_processed += 1

    all_scanned_warnings: list[str] = []

    # A PDF is authoritative only when its exact SHA-256 is explicitly allowlisted.
    # Filename or absence of the word SAMPLE is not sufficient proof of authenticity.
    settings = get_settings()
    try:
        authorized_hashes = json.loads(settings.regulatory_authorized_sha256_json or "{}")
        if not isinstance(authorized_hashes, dict):
            raise ValueError("REGULATORY_AUTHORIZED_SHA256_JSON must be a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid regulatory source allowlist: {exc}") from exc

    for filepath in sorted(regulations_dir.glob("*.pdf")):
        logger.info("ingesting_pdf_file", file=filepath.name)
        file_digest = hashlib.sha256(filepath.read_bytes()).hexdigest()
        allowlisted_digest = str(authorized_hashes.get(filepath.name, "")).lower()
        is_authoritative = bool(allowlisted_digest) and hmac.compare_digest(file_digest, allowlisted_digest)
        if not is_authoritative:
            logger.warning("regulatory_pdf_not_allowlisted_as_authoritative", file=filepath.name, sha256=file_digest)
        pages, scanned_warnings = _extract_text_from_pdf(filepath)
        all_scanned_warnings.extend(scanned_warnings)
        for page_num, page_text in pages:
            # Unverified documents and pages explicitly labelled SAMPLE remain sample evidence.
            is_sample = not is_authoritative or "SAMPLE" in page_text.upper()[:200]
            page_chunks = _chunk_document(
                page_text,
                source_file=filepath.name,
                page_number=page_num,
                is_sample=is_sample,
            )
            all_chunks.extend(page_chunks)
        if pages:
            files_processed += 1

    if not all_chunks:
        logger.warning("no_chunks_to_ingest")
        return {"files": 0, "chunks": 0, "scanned_page_warnings": all_scanned_warnings}

    # Embed all chunks
    texts = [c["text"] for c in all_chunks]
    embeddings = embedding_provider.embed_texts(texts)

    # Add to ChromaDB in batches
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=batch_embeddings,
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )

    stats = {
        "files": files_processed,
        "chunks": len(all_chunks),
        "collection": COLLECTION_NAME,
        "authoritative_files_allowlisted": sum(
            1 for path in regulations_dir.glob("*.pdf")
            if str(authorized_hashes.get(path.name, "")).lower()
            and hmac.compare_digest(hashlib.sha256(path.read_bytes()).hexdigest(), str(authorized_hashes.get(path.name, "")).lower())
        ),
        "scanned_page_warnings": all_scanned_warnings,
    }
    logger.info("ingestion_complete", **stats)
    return stats
