"""Offline tests (no API key or database needed): extraction, cleaning, chunking, file errors.

Run with:  python -m unittest discover tests
"""

import tempfile
import unittest
from pathlib import Path

from docsearch.chunking import chunk_text, fixed_size_chunks, paragraph_chunks, sentence_chunks
from docsearch.errors import EmptyDocumentError, MissingFileError, UnsupportedFileTypeError
from docsearch.extract import clean_text, extract_text

DOCS = Path(__file__).resolve().parent.parent / "docs"
SAMPLE = "First sentence here. Second one follows! Is this the third? Yes.\n\nA new paragraph starts here and goes on for a while, long enough to count as a real paragraph."


class ExtractionTests(unittest.TestCase):
    def test_pdf_and_docx_have_text(self):
        for name in ("support_knowledge_base.pdf", "support_knowledge_base.docx"):
            text = extract_text(DOCS / name)
            self.assertIn("Refunds are available", text)

    def test_missing_file(self):
        with self.assertRaises(MissingFileError):
            extract_text("does/not/exist.pdf")

    def test_unsupported_type(self):
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            with self.assertRaises(UnsupportedFileTypeError):
                extract_text(f.name)

    def test_corrupted_pdf(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(b"this is not a pdf")
            f.flush()
            with self.assertRaises(UnsupportedFileTypeError):
                extract_text(f.name)

    def test_pdf_without_text(self):
        from pypdf import PdfWriter

        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            writer = PdfWriter()
            writer.add_blank_page(width=200, height=200)
            writer.write(f)
            f.flush()
            with self.assertRaises(EmptyDocumentError):
                extract_text(f.name)

    def test_clean_text(self):
        self.assertEqual(clean_text("hel-\nlo   world\n\n\n\nnext\x00"), "hello world\n\nnext")


class ChunkingTests(unittest.TestCase):
    def test_fixed_overlap(self):
        text = " ".join(f"word{i}" for i in range(300))
        chunks = fixed_size_chunks(text, chunk_size=200, overlap=50)
        self.assertTrue(all(len(c) <= 200 for c in chunks))
        # consecutive chunks share text (the overlap)
        self.assertIn(chunks[0].split()[-1], chunks[1])

    def test_sentence_chunks_keep_sentences_whole(self):
        chunks = sentence_chunks(SAMPLE, max_chars=30)
        self.assertEqual(chunks[0], "First sentence here.")
        self.assertTrue(all(c.endswith((".", "!", "?")) for c in chunks))

    def test_paragraph_chunks(self):
        chunks = paragraph_chunks(SAMPLE, max_chars=500)
        self.assertEqual(len(chunks), 2)

    def test_unknown_strategy(self):
        with self.assertRaises(ValueError):
            chunk_text(SAMPLE, "words")


if __name__ == "__main__":
    unittest.main()
