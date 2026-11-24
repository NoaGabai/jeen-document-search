"""Semantic search over indexed documents.

Usage:
    python search.py --query "login issue"
    python search.py --query "refund policy" --top-k 3 --strategy sentence
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from docsearch.chunking import STRATEGIES
from docsearch.config import load_settings
from docsearch.errors import DocSearchError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search indexed documents by meaning.")
    parser.add_argument("--query", required=True, help="What you are looking for, in natural language")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return (default: 5)")
    parser.add_argument("--strategy", choices=STRATEGIES, help="Only search chunks created with this strategy")
    parser.add_argument("--file", dest="filename", help="Only search chunks from this file name")
    parser.add_argument("--min-score", type=float, default=0.0, help="Hide results below this similarity (0-1, default: 0)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.query.strip():
        print("Error: --query cannot be empty.", file=sys.stderr)
        return 2
    try:
        settings = load_settings()

        from docsearch.db import VectorStore
        from docsearch.embeddings import GeminiEmbedder

        query_vector = GeminiEmbedder(settings.gemini_api_key).embed_query(args.query)
        with VectorStore(settings.postgres_url) as store:
            results = store.search(query_vector, top_k=args.top_k, strategy=args.strategy, filename=args.filename)
        results = [r for r in results if r.similarity >= args.min_score]

        if not results:
            print(f'No results found for "{args.query}".')
            print("Tip: index a document first, remove filters, or lower --min-score.")
            return 0

        print(f'Top {len(results)} results for "{args.query}":\n')
        for rank, r in enumerate(results, start=1):
            print(f"#{rank}  similarity={r.similarity:.3f}  file={r.filename}  strategy={r.split_strategy}  chunk={r.chunk_index}  id={r.id}")
            print(textwrap.indent(textwrap.fill(r.chunk_text, width=100), "    "))
            print()
        return 0

    except DocSearchError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
