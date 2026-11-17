#!/usr/bin/env python
"""
CLI script to ingest FSSAI regulation documents into the ChromaDB vector store.

Usage:
    python scripts/ingest_regulations.py
    python scripts/ingest_regulations.py --mock-embeddings  # use hash embeddings (fast, offline)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import REGULATIONS_DIR
from app.rag.ingestion import ingest_regulations


def main():
    parser = argparse.ArgumentParser(description="Ingest FSSAI regulation documents into ChromaDB.")
    parser.add_argument(
        "--regulations-dir",
        type=str,
        default=str(REGULATIONS_DIR),
        help="Directory containing regulation text files.",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=None,
        help="ChromaDB persistence directory (default: from settings).",
    )
    parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help="Use hash-based embeddings (fast, no ML model required).",
    )
    args = parser.parse_args()

    print(f"Ingesting regulations from: {args.regulations_dir}")
    stats = ingest_regulations(
        regulations_dir=args.regulations_dir,
        persist_dir=args.persist_dir,
        use_mock_embeddings=args.mock_embeddings,
    )
    print(f"Done! Files: {stats['files']}, Chunks: {stats['chunks']}")


if __name__ == "__main__":
    main()
