"""PostgreSQL + pgvector storage and similarity search."""

from __future__ import annotations

from dataclasses import dataclass

from .config import EMBEDDING_DIM, TABLE_NAME
from .errors import DatabaseError

SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id             BIGSERIAL PRIMARY KEY,
    chunk_text     TEXT        NOT NULL,
    embedding      vector({EMBEDDING_DIM}) NOT NULL,
    filename       TEXT        NOT NULL,
    split_strategy TEXT        NOT NULL,
    chunk_index    INTEGER     NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- HNSW index for fast cosine-similarity search.
CREATE INDEX IF NOT EXISTS {TABLE_NAME}_embedding_idx
    ON {TABLE_NAME} USING hnsw (embedding vector_cosine_ops);
"""


@dataclass
class SearchResult:
    id: int
    filename: str
    split_strategy: str
    chunk_index: int
    chunk_text: str
    similarity: float


def _to_vector_literal(vector: list[float]) -> str:
    """pgvector accepts vectors as text like '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.7f}" for x in vector) + "]"


class VectorStore:
    def __init__(self, postgres_url: str):
        import psycopg

        try:
            self._conn = psycopg.connect(postgres_url, connect_timeout=10)
        except psycopg.Error as exc:
            # Do not print the exception text: it can contain the host and user name.
            raise DatabaseError(
                "Could not connect to PostgreSQL. Check POSTGRES_URL in your .env file "
                f"and that the database is running ({type(exc).__name__})."
            ) from exc

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def ensure_schema(self) -> None:
        self._run(lambda cur: cur.execute(SCHEMA_SQL), "create the table")

    def replace_chunks(self, filename: str, strategy: str, chunks: list[str], embeddings: list[list[float]]) -> int:
        """Store chunks for (filename, strategy) in one transaction.

        Existing rows for the same file and strategy are replaced, so re-indexing a
        document does not create duplicates.
        """

        def work(cur):
            cur.execute(f"DELETE FROM {TABLE_NAME} WHERE filename = %s AND split_strategy = %s", (filename, strategy))
            cur.executemany(
                f"INSERT INTO {TABLE_NAME} (chunk_text, embedding, filename, split_strategy, chunk_index) "
                "VALUES (%s, %s::vector, %s, %s, %s)",
                [(text, _to_vector_literal(vec), filename, strategy, i) for i, (text, vec) in enumerate(zip(chunks, embeddings))],
            )

        self._run(work, "save the chunks")
        return len(chunks)

    def search(self, query_embedding: list[float], top_k: int = 5, strategy: str | None = None, filename: str | None = None) -> list[SearchResult]:
        """Return the top_k chunks closest to the query (cosine similarity, 1 = identical)."""
        vec = _to_vector_literal(query_embedding)
        where, params = [], []
        if strategy:
            where.append("split_strategy = %s")
            params.append(strategy)
        if filename:
            where.append("filename = %s")
            params.append(filename)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        sql = (
            f"SELECT id, filename, split_strategy, chunk_index, chunk_text, "
            f"1 - (embedding <=> %s::vector) AS similarity "
            f"FROM {TABLE_NAME} {where_sql} "
            f"ORDER BY embedding <=> %s::vector LIMIT %s"
        )
        rows: list = []

        def work(cur):
            cur.execute(sql, [vec, *params, vec, top_k])
            rows.extend(cur.fetchall())

        self._run(work, "search")
        return [SearchResult(*row) for row in rows]

    def _run(self, work, action: str) -> None:
        import psycopg

        try:
            with self._conn.transaction():
                with self._conn.cursor() as cur:
                    work(cur)
        except psycopg.errors.UndefinedTable as exc:
            raise DatabaseError(f"Table '{TABLE_NAME}' does not exist yet. Index a document first.") from exc
        except psycopg.Error as exc:
            raise DatabaseError(f"Database error while trying to {action} ({type(exc).__name__}).") from exc
