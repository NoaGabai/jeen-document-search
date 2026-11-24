"""Three chunking strategies: fixed-size with overlap, sentence-based, paragraph-based."""

from __future__ import annotations

import re

STRATEGIES = ("fixed", "sentence", "paragraph")

DEFAULT_CHUNK_SIZE = 800  # characters
DEFAULT_OVERLAP = 150  # characters

# A sentence ends with . ! or ? followed by whitespace and a new sentence start.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9֐-׿])")


def chunk_text(text: str, strategy: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    """Split text into chunks using the chosen strategy."""
    if strategy == "fixed":
        chunks = fixed_size_chunks(text, chunk_size, overlap)
    elif strategy == "sentence":
        chunks = sentence_chunks(text, chunk_size)
    elif strategy == "paragraph":
        chunks = paragraph_chunks(text, chunk_size)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose one of: {', '.join(STRATEGIES)}")
    return [c.strip() for c in chunks if c.strip()]


def fixed_size_chunks(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    """Windows of ~chunk_size characters; each window repeats the last `overlap` characters
    of the previous one so that context is not lost at the boundary.
    Cuts are moved back to the nearest space so words are not split in half."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    text = re.sub(r"\s+", " ", text).strip()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + chunk_size // 2, end)
            if space != -1:
                end = space
        chunks.append(text[start:end])
        if end >= len(text):
            break
        next_start = end - overlap
        # Start the next chunk at a word boundary too.
        space = text.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start
    return chunks


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for paragraph in text.split("\n\n"):
        sentences.extend(s.strip() for s in _SENTENCE_BOUNDARY.split(paragraph.replace("\n", " ")) if s.strip())
    return sentences


def sentence_chunks(text: str, max_chars: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    """Group whole sentences into chunks of up to max_chars. A sentence is never cut,
    unless a single sentence is itself longer than max_chars."""
    return _pack(split_sentences(text), max_chars, joiner=" ")


def paragraph_chunks(text: str, max_chars: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    """One chunk per paragraph. Headings (short lines without final punctuation) are
    merged with the next paragraph; paragraphs longer than max_chars fall back to
    sentence chunking."""
    chunks: list[str] = []
    pending = ""
    for paragraph in (p.strip() for p in text.split("\n\n")):
        if not paragraph:
            continue
        if pending:
            paragraph = f"{pending}\n{paragraph}"
            pending = ""
        is_heading = len(paragraph) < 80 and not paragraph.endswith((".", "!", "?"))
        if is_heading:  # attach headings to the paragraph that follows them
            pending = paragraph
            continue
        if len(paragraph) > max_chars:
            chunks.extend(sentence_chunks(paragraph, max_chars))
        else:
            chunks.append(paragraph)
    if pending:
        if chunks and len(chunks[-1]) + len(pending) <= max_chars:
            chunks[-1] = f"{chunks[-1]}\n{pending}"
        else:
            chunks.append(pending)
    return chunks


def _pack(pieces: list[str], max_chars: int, joiner: str) -> list[str]:
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}{joiner}{piece}" if current else piece
        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks
