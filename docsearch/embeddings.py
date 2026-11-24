"""Embeddings with the Gemini API (gemini-embedding-001).

This is the only file that talks to Gemini. To run in an air-gapped environment,
replace this module with one that calls a locally hosted embedding model and returns
vectors of the same size.
"""

from __future__ import annotations

import time

from .config import EMBEDDING_DIM, EMBEDDING_MODEL
from .errors import EmbeddingError

BATCH_SIZE = 50  # texts per API call
MAX_RETRIES = 3


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str = EMBEDDING_MODEL, dim: int = EMBEDDING_DIM):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunks for storage (task type RETRIEVAL_DOCUMENT)."""
        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            vectors.extend(self._embed(texts[i : i + BATCH_SIZE], "RETRIEVAL_DOCUMENT"))
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query (task type RETRIEVAL_QUERY)."""
        return self._embed([query], "RETRIEVAL_QUERY")[0]

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import errors, types

        config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim)
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = self._client.models.embed_content(model=self.model, contents=texts, config=config)
                vectors = [list(e.values) for e in result.embeddings]
                if len(vectors) != len(texts) or any(len(v) != self.dim for v in vectors):
                    raise EmbeddingError("Gemini returned an unexpected number or size of embeddings.")
                return vectors
            except errors.APIError as exc:
                retryable = exc.code in (429, 500, 502, 503, 504)
                if retryable and attempt < MAX_RETRIES:
                    time.sleep(2**attempt)  # 2s, 4s ... back off on rate limits / server errors
                    continue
                hint = " Check that GEMINI_API_KEY is valid." if exc.code in (400, 401, 403) else ""
                raise EmbeddingError(f"Gemini embedding request failed (HTTP {exc.code}).{hint}") from exc
            except EmbeddingError:
                raise
            except Exception as exc:  # network errors, timeouts
                if attempt < MAX_RETRIES:
                    time.sleep(2**attempt)
                    continue
                raise EmbeddingError(f"Could not reach the Gemini API ({type(exc).__name__}).") from exc
        raise EmbeddingError("Gemini embedding request failed.")  # pragma: no cover
