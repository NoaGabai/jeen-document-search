"""Configuration loaded from environment variables / a local .env file.

Secrets (API key, database URL) are never hard-coded and never printed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from .errors import ConfigError

# Embedding settings. gemini-embedding-001 returns 3072 dimensions by default;
# we request 768, which keeps storage small, allows an HNSW index in pgvector
# (limited to 2000 dims) and loses very little quality.
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768

TABLE_NAME = "document_chunks"


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    postgres_url: str


def load_settings() -> Settings:
    """Read settings from the environment (and .env). Raise ConfigError if anything is missing."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    db_url = os.getenv("POSTGRES_URL", "").strip()

    missing = [name for name, value in (("GEMINI_API_KEY", api_key), ("POSTGRES_URL", db_url)) if not value]
    if missing:
        raise ConfigError(
            f"Missing environment variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill in the values."
        )
    return Settings(gemini_api_key=api_key, postgres_url=db_url)
