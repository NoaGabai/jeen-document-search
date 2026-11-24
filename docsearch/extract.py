"""Text extraction from PDF and DOCX files, plus light cleaning.

Output format: plain text where paragraphs are separated by a blank line ("\\n\\n").
The paragraph chunking strategy relies on that convention.
"""

from __future__ import annotations

import re
from pathlib import Path

from .errors import EmptyDocumentError, MissingFileError, UnsupportedFileTypeError

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def extract_text(path: str | Path) -> str:
    """Return clean text from a PDF or DOCX file.

    Raises MissingFileError, UnsupportedFileTypeError or EmptyDocumentError.
    """
    path = Path(path)
    if not path.is_file():
        raise MissingFileError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix or '(none)'}'. Supported types: PDF, DOCX."
        )

    try:
        raw = _extract_pdf(path) if suffix == ".pdf" else _extract_docx(path)
    except (UnsupportedFileTypeError, EmptyDocumentError):
        raise
    except Exception as exc:  # corrupted / password-protected / not really a PDF or DOCX
        raise UnsupportedFileTypeError(f"Could not read '{path.name}': the file may be corrupted or protected ({type(exc).__name__}).") from exc

    text = clean_text(raw)
    if not text:
        raise EmptyDocumentError(
            f"No extractable text found in '{path.name}'. "
            "If it is a scanned document, it needs OCR first."
        )
    return text


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")  # many "encrypted" PDFs open with an empty password
        except Exception as exc:
            raise UnsupportedFileTypeError(f"'{path.name}' is password-protected.") from exc

    pages = [page.extract_text() or "" for page in reader.pages]
    # Treat each page break as a paragraph break.
    return "\n\n".join(_pdf_lines_to_paragraphs(p) for p in pages)


def _pdf_lines_to_paragraphs(page_text: str) -> str:
    """PDF text arrives as visual lines. Re-join lines into paragraphs.

    A line ends a paragraph when it is followed by a blank line, or when it ends with
    sentence punctuation and the next line starts like a new block (capital letter,
    bullet or number) and the current line is noticeably shorter than a full line.
    """
    lines = [ln.rstrip() for ln in page_text.splitlines()]
    if not lines:
        return ""
    full_width = max((len(ln) for ln in lines), default=0)

    paragraphs: list[str] = []
    current: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(stripped)

        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        ends_sentence = stripped.endswith((".", "!", "?", ":"))
        short_line = len(stripped) < 0.75 * full_width
        next_starts_block = bool(re.match(r"^([A-Z0-9]|[-•*]\s)", nxt))
        is_heading = len(stripped) < 60 and not stripped.endswith((".", ",", ";")) and next_starts_block and short_line
        if nxt and ((ends_sentence and short_line and next_starts_block) or is_heading):
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def _extract_docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    blocks = [p.text for p in document.paragraphs]
    # Tables are common in business documents; keep their text, one row per line.
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))
    return "\n\n".join(blocks)


def clean_text(text: str) -> str:
    """Normalize whitespace and remove common extraction artifacts."""
    text = text.replace("\x00", "").replace("­", "")  # null bytes, soft hyphens
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # words hyphenated across lines
    text = re.sub(r"[ \t ]+", " ", text)  # collapse spaces/tabs
    text = re.sub(r" *\n *", "\n", text)  # trim spaces around newlines
    text = re.sub(r"\n{3,}", "\n\n", text)  # at most one blank line
    # Drop empty paragraphs.
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return "\n\n".join(p for p in paragraphs if p)
