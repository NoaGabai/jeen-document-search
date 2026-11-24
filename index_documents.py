"""Index a PDF or DOCX file: extract text -> chunk -> embed with Gemini -> store in PostgreSQL.

Usage:
    python index_documents.py --file ./docs/example.pdf --strategy paragraph
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from docsearch.chunking import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, STRATEGIES, chunk_text
from docsearch.config import load_settings
from docsearch.errors import DocSearchError
from docsearch.extract import extract_text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index a PDF/DOCX document for semantic search.")
    parser.add_argument("--file", required=True, help="Path to a .pdf or .docx file")
    parser.add_argument("--strategy", choices=STRATEGIES, default="paragraph", help="Chunking strategy (default: paragraph)")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help=f"Max characters per chunk (default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help=f"Overlap in characters for the fixed strategy (default: {DEFAULT_OVERLAP})")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    try:
        # Validate configuration and the file before spending any API calls.
        settings = load_settings()
        path = Path(args.file)

        text = extract_text(path)
        print(f"[1/4] Extracted {len(text):,} characters from {path.name}")

        chunks = chunk_text(text, args.strategy, args.chunk_size, args.overlap)
        sizes = [len(c) for c in chunks]
        print(f"[2/4] Split into {len(chunks)} chunks using '{args.strategy}' strategy "
              f"(avg {sum(sizes) // len(sizes)} chars, min {min(sizes)}, max {max(sizes)})")

        from docsearch.db import VectorStore
        from docsearch.embeddings import GeminiEmbedder

        # Connect to the database first, so a DB problem does not waste embedding API calls.
        with VectorStore(settings.postgres_url) as store:
            store.ensure_schema()

            embeddings = GeminiEmbedder(settings.gemini_api_key).embed_documents(chunks)
            print(f"[3/4] Created {len(embeddings)} embeddings with gemini-embedding-001 ({len(embeddings[0])} dimensions)")

            saved = store.replace_chunks(path.name, args.strategy, chunks, embeddings)
        print(f"[4/4] Stored {saved} chunks in PostgreSQL (table: document_chunks)")
        print(f"Done in {time.perf_counter() - started:.1f}s")
        return 0

    except DocSearchError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:  # invalid chunking parameters
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
