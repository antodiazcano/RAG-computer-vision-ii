"""
Tests for src/embed_documents/save_files.py
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.embed_documents.save_files import (
    _build_corpus_index,
    delete_vectors_by_metadata,
)


class TestDeleteVectorsByMetadata:
    """Tests for the delete_vectors_by_metadata function."""

    def test_calls_delete_with_filter(self) -> None:
        """Checks that the Pinecone index delete is called with the given filter."""
        index = MagicMock()
        metadata_filter = {"source": "file.pdf"}

        with (
            patch(
                "src.embed_documents.save_files.load_registry",
                return_value={"file.pdf": "h1"},
            ),
            patch("src.embed_documents.save_files.save_registry") as _,
        ):
            delete_vectors_by_metadata(index, metadata_filter)

        index.delete.assert_called_once_with(filter={"source": "file.pdf"})

    def test_removes_source_from_registry(self) -> None:
        """Checks that the matching source is removed from the registry."""
        index = MagicMock()
        reg = {"file.pdf": "h1", "other.tex": "h2"}

        with (
            patch("src.embed_documents.save_files.load_registry", return_value=reg),
            patch("src.embed_documents.save_files.save_registry") as mock_save,
        ):
            delete_vectors_by_metadata(index, {"source": "file.pdf"})

        mock_save.assert_called_once_with({"other.tex": "h2"})

    def test_removes_by_doc_type_from_registry(self) -> None:
        """Checks that all files matching the doc_type extension are removed."""
        index = MagicMock()
        reg = {"ch1.tex": "h1", "ch2.tex": "h2", "slides.pdf": "h3"}

        with (
            patch("src.embed_documents.save_files.load_registry", return_value=reg),
            patch("src.embed_documents.save_files.save_registry") as mock_save,
        ):
            delete_vectors_by_metadata(index, {"doc_type": "tex"})

        mock_save.assert_called_once_with({"slides.pdf": "h3"})


class TestBuildCorpusIndex:
    """Tests for the _build_corpus_index function."""

    def test_extracts_toc_from_tex_files(self, tmp_path: Path) -> None:
        """Checks that .tex files produce a toc entry."""
        tex = tmp_path / "ch1.tex"
        tex.write_text(
            "\\begin{document}\n\\chapter{Intro}\n\\section{S1}\nText.\n"
            "\\end{document}\n",
            encoding="utf-8",
        )

        with patch("src.embed_documents.save_files.config") as mock_config:
            mock_config.paths.documents_folder = tmp_path
            result = _build_corpus_index()

        assert len(result) == 1
        assert result[0]["source"] == "ch1.tex"
        toc = result[0]["toc"]
        assert isinstance(toc, list)
        assert len(toc) == 2
        assert toc[0]["name"] == "Intro"
        assert toc[1]["name"] == "S1"

    def test_pdf_files_have_empty_toc(self, tmp_path: Path) -> None:
        """Checks that .pdf files produce an entry with an empty toc."""
        pdf = tmp_path / "slides.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        with patch("src.embed_documents.save_files.config") as mock_config:
            mock_config.paths.documents_folder = tmp_path
            result = _build_corpus_index()

        assert len(result) == 1
        assert result[0]["source"] == "slides.pdf"
        assert result[0]["toc"] == []

    def test_mixed_files(self, tmp_path: Path) -> None:
        """Checks that both .tex and .pdf files are included."""
        tex = tmp_path / "ch1.tex"
        tex.write_text(
            "\\begin{document}\n\\chapter{Ch1}\nText.\n\\end{document}\n",
            encoding="utf-8",
        )
        pdf = tmp_path / "slides.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        with patch("src.embed_documents.save_files.config") as mock_config:
            mock_config.paths.documents_folder = tmp_path
            result = _build_corpus_index()

        sources = [r["source"] for r in result]
        assert "ch1.tex" in sources
        assert "slides.pdf" in sources

    def test_empty_folder(self, tmp_path: Path) -> None:
        """Checks that an empty folder returns an empty list."""
        with patch("src.embed_documents.save_files.config") as mock_config:
            mock_config.paths.documents_folder = tmp_path
            result = _build_corpus_index()

        assert not result
