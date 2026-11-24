"""Custom exceptions.

Each failure the assignment asks us to handle has its own exception type, so the
command-line scripts can show a clear, friendly message instead of a stack trace.
"""


class DocSearchError(Exception):
    """Base class for all expected (handled) errors in this project."""


class ConfigError(DocSearchError):
    """A required environment variable is missing."""


class MissingFileError(DocSearchError):
    """The input file does not exist."""


class UnsupportedFileTypeError(DocSearchError):
    """The input file is not a PDF or DOCX."""


class EmptyDocumentError(DocSearchError):
    """The document contains no extractable text (e.g. a scanned PDF)."""


class EmbeddingError(DocSearchError):
    """The Gemini embedding API call failed."""


class DatabaseError(DocSearchError):
    """Connecting to or querying PostgreSQL failed."""
