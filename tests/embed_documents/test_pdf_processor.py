"""
Tests for src/embed_documents/pdf_processor.py
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.embed_documents.pdf_processor import PDFProcessor


def _mock_fitz_doc(pages: list[str]) -> MagicMock:
    """Creates a mock fitz document with the given page texts.

    Args:
        pages: List of text strings, one per page. Empty string simulates a
            page with no extractable text.

    Returns:
        A MagicMock that behaves like a fitz.Document used as a context manager.
    """
    doc = MagicMock()
    doc.__len__ = lambda _: len(pages)
    mock_pages = []
    for text in pages:
        page = MagicMock()
        page.get_text.return_value = text
        mock_pages.append(page)
    doc.__getitem__ = lambda _, i: mock_pages[i]
    doc.__enter__ = lambda _: doc
    doc.__exit__ = lambda *_: None
    return doc


class TestObtainChunks:
    """Tests for the PDFProcessor._obtain_chunks method."""

    @patch("src.embed_documents.pdf_processor.fitz.open")
    def test_extracts_chunks_from_all_pages(self, mock_fitz_open: MagicMock) -> None:
        """Checks that a chunk is created for each page with text."""
        mock_fitz_open.return_value = _mock_fitz_doc(["page one", "page two"])

        proc = PDFProcessor(Path("doc.pdf"), MagicMock(), MagicMock())
        chunks = proc._obtain_chunks()

        assert len(chunks) == 2
        assert chunks[0]["text"] == "page one"
        assert chunks[0]["location"] == 1
        assert chunks[0]["total_locations"] == 2
        assert chunks[1]["text"] == "page two"
        assert chunks[1]["location"] == 2

    @patch("src.embed_documents.pdf_processor.fitz.open")
    def test_skips_empty_pages(self, mock_fitz_open: MagicMock) -> None:
        """Checks that pages with no extractable text are skipped."""
        mock_fitz_open.return_value = _mock_fitz_doc(["text", "", "more text"])

        proc = PDFProcessor(Path("doc.pdf"), MagicMock(), MagicMock())
        chunks = proc._obtain_chunks()

        assert len(chunks) == 2
        assert chunks[0]["location"] == 1
        assert chunks[1]["location"] == 3

    @patch("src.embed_documents.pdf_processor.fitz.open")
    def test_skips_whitespace_only_pages(self, mock_fitz_open: MagicMock) -> None:
        """Checks that pages with only whitespace are skipped."""
        mock_fitz_open.return_value = _mock_fitz_doc(["   \n\t  "])

        proc = PDFProcessor(Path("doc.pdf"), MagicMock(), MagicMock())
        chunks = proc._obtain_chunks()

        assert chunks == []

    @patch("src.embed_documents.pdf_processor.fitz.open")
    def test_sets_correct_metadata(self, mock_fitz_open: MagicMock) -> None:
        """Checks that source filename and doc_type are set correctly."""
        mock_fitz_open.return_value = _mock_fitz_doc(["content"])

        proc = PDFProcessor(Path("/some/path/lecture.pdf"), MagicMock(), MagicMock())
        chunks = proc._obtain_chunks()

        assert chunks[0]["source"] == "lecture.pdf"
        assert chunks[0]["doc_type"] == "pdf"

    @patch("src.embed_documents.pdf_processor.fitz.open")
    def test_empty_document(self, mock_fitz_open: MagicMock) -> None:
        """Checks that a document with no pages returns an empty list."""
        mock_fitz_open.return_value = _mock_fitz_doc([])

        proc = PDFProcessor(Path("empty.pdf"), MagicMock(), MagicMock())
        chunks = proc._obtain_chunks()

        assert chunks == []
