"""
Tests for src/save_docs_to_db.py
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.config import config


# Mock the module-level clients before importing
with (
    patch("src.utils.get_gen_ai_client", return_value=MagicMock()),
    patch("src.utils.get_index_vector_db", return_value=MagicMock()),
):
    from src.save_docs_to_db import (
        GEN_AI_CLIENT,
        PINECONE_IDX,
        _embed_text,
        _obtain_chunks,
        _save_file_as_embeddings,
        save_all_files,
    )


# Helpers


def _mock_fitz_doc(pages: list[str]) -> MagicMock:
    """Creates a mock fitz document with the given page texts.

    Args:
        pages: List of text strings, one per page. Empty string simulates a
            page with no extractable text.

    Returns:
        A MagicMock that behaves like a fitz.Document.
    """
    doc = MagicMock()
    doc.__len__ = lambda _: len(pages)
    mock_pages = []
    for text in pages:
        page = MagicMock()
        page.get_text.return_value = text
        mock_pages.append(page)
    doc.__getitem__ = lambda _, i: mock_pages[i]
    return doc


def _setup_embed_mock(values: list[float] | None = None) -> None:
    """Sets up the GEN_AI_CLIENT embed mock with default or custom values.

    Args:
        values: Embedding values to return. Defaults to [0.1, 0.2].
    """
    if values is None:
        values = [0.1, 0.2]
    mock_embedding = MagicMock()
    mock_embedding.values = values
    mock_response = MagicMock()
    mock_response.embeddings = [mock_embedding]
    GEN_AI_CLIENT.models.embed_content.return_value = mock_response


class TestEmbedText:
    """Tests for the _embed_text function."""

    def test_returns_embedding_values(self) -> None:
        """Checks that the embedding values from the API response are returned."""
        _setup_embed_mock([0.5, 0.6, 0.7])

        result = _embed_text("some text")

        assert result == [0.5, 0.6, 0.7]

    def test_passes_correct_model_and_task_type(self) -> None:
        """Checks that the correct model and RETRIEVAL_DOCUMENT task type are used."""
        _setup_embed_mock()

        _embed_text("hello")

        call_kwargs = GEN_AI_CLIENT.models.embed_content.call_args
        assert call_kwargs.kwargs["model"] == config.embedding_model.embedding_model
        assert call_kwargs.kwargs["contents"] == "hello"


class TestObtainChunks:
    """Tests for the _obtain_chunks function."""

    @patch("src.save_docs_to_db.fitz.open")
    def test_extracts_chunks_from_all_pages(self, mock_fitz_open: MagicMock) -> None:
        """Checks that a chunk is created for each page with text."""
        mock_fitz_open.return_value = _mock_fitz_doc(["page one", "page two"])

        chunks = _obtain_chunks(Path("doc.pdf"))

        assert len(chunks) == 2
        assert chunks[0]["text"] == "page one"
        assert chunks[0]["page"] == 1
        assert chunks[0]["total_pages"] == 2
        assert chunks[1]["text"] == "page two"
        assert chunks[1]["page"] == 2

    @patch("src.save_docs_to_db.fitz.open")
    def test_skips_empty_pages(self, mock_fitz_open: MagicMock) -> None:
        """Checks that pages with no extractable text are skipped."""
        mock_fitz_open.return_value = _mock_fitz_doc(["text", "", "more text"])

        chunks = _obtain_chunks(Path("doc.pdf"))

        assert len(chunks) == 2
        assert chunks[0]["page"] == 1
        assert chunks[1]["page"] == 3

    @patch("src.save_docs_to_db.fitz.open")
    def test_skips_whitespace_only_pages(self, mock_fitz_open: MagicMock) -> None:
        """Checks that pages with only whitespace are skipped."""
        mock_fitz_open.return_value = _mock_fitz_doc(["   \n\t  "])

        chunks = _obtain_chunks(Path("doc.pdf"))

        assert chunks == []

    @patch("src.save_docs_to_db.fitz.open")
    def test_sets_correct_metadata(self, mock_fitz_open: MagicMock) -> None:
        """Checks that source filename and doc_type are set correctly."""
        mock_fitz_open.return_value = _mock_fitz_doc(["content"])

        chunks = _obtain_chunks(Path("/some/path/lecture.pdf"))

        assert chunks[0]["source"] == "lecture.pdf"
        assert chunks[0]["doc_type"] == "pdf"

    @patch("src.save_docs_to_db.fitz.open")
    def test_empty_document(self, mock_fitz_open: MagicMock) -> None:
        """Checks that a document with no pages returns an empty list."""
        mock_fitz_open.return_value = _mock_fitz_doc([])

        chunks = _obtain_chunks(Path("empty.pdf"))

        assert chunks == []


class TestSaveFileAsEmbeddings:
    """Tests for the _save_file_as_embeddings function."""

    def setup_method(self) -> None:
        """Resets mocks before each test."""
        GEN_AI_CLIENT.models.embed_content.reset_mock()
        PINECONE_IDX.upsert.reset_mock()
        _setup_embed_mock()

    @patch("src.save_docs_to_db._obtain_chunks")
    def test_returns_chunk_count(self, mock_obtain: MagicMock) -> None:
        """Checks that the number of chunks is returned."""
        mock_obtain.return_value = [
            {
                "text": "t1",
                "source": "a.pdf",
                "page": 1,
                "total_pages": 1,
                "doc_type": "pdf",
            },
            {
                "text": "t2",
                "source": "a.pdf",
                "page": 2,
                "total_pages": 2,
                "doc_type": "pdf",
            },
        ]

        result = _save_file_as_embeddings(Path("a.pdf"))

        assert result == 2

    @patch("src.save_docs_to_db._obtain_chunks")
    def test_upserts_to_pinecone(self, mock_obtain: MagicMock) -> None:
        """Checks that vectors are upserted to Pinecone."""
        mock_obtain.return_value = [
            {
                "text": "t",
                "source": "a.pdf",
                "page": 1,
                "total_pages": 1,
                "doc_type": "pdf",
            },
        ]

        _save_file_as_embeddings(Path("a.pdf"))

        PINECONE_IDX.upsert.assert_called_once()
        vectors = PINECONE_IDX.upsert.call_args.kwargs["vectors"]
        assert len(vectors) == 1
        assert vectors[0]["values"] == [0.1, 0.2]
        assert vectors[0]["metadata"]["text"] == "t"

    @patch("src.save_docs_to_db._obtain_chunks")
    def test_batches_large_uploads(self, mock_obtain: MagicMock) -> None:
        """Checks that vectors are upserted in batches of 50."""
        chunks = [
            {
                "text": f"t{i}",
                "source": "a.pdf",
                "page": i,
                "total_pages": 120,
                "doc_type": "pdf",
            }
            for i in range(1, 121)
        ]
        mock_obtain.return_value = chunks

        _save_file_as_embeddings(Path("a.pdf"))

        assert PINECONE_IDX.upsert.call_count == 3  # 50 + 50 + 20

    @patch("src.save_docs_to_db._obtain_chunks")
    def test_no_chunks_no_upsert(self, mock_obtain: MagicMock) -> None:
        """Checks that no upsert is called when there are no chunks."""
        mock_obtain.return_value = []

        result = _save_file_as_embeddings(Path("empty.pdf"))

        assert result == 0
        PINECONE_IDX.upsert.assert_not_called()


class TestSaveAllFiles:
    """Tests for the save_all_files function."""

    @patch("src.save_docs_to_db.save_registry")
    @patch("src.save_docs_to_db.file_hash")
    @patch("src.save_docs_to_db.load_registry")
    @patch("src.save_docs_to_db._save_file_as_embeddings")
    def test_indexes_new_files(
        self,
        mock_save_file: MagicMock,
        mock_load_reg: MagicMock,
        mock_hash: MagicMock,
        mock_save_reg: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Checks that new (unindexed) files are indexed and the registry is updated."""
        pdf = tmp_path / "new.pdf"
        pdf.write_bytes(b"%PDF-fake")

        with (
            patch.object(config.paths, "documents_folder", tmp_path),
            patch.object(config.paths, "supported_extensions", {".pdf"}),
        ):
            mock_load_reg.return_value = {}
            mock_hash.return_value = "hash_new"
            mock_save_file.return_value = 5

            save_all_files()

        mock_save_file.assert_called_once_with(pdf)
        mock_save_reg.assert_called_once()
        saved_reg = mock_save_reg.call_args[0][0]
        assert saved_reg["new.pdf"] == "hash_new"

    @patch("src.save_docs_to_db.save_registry")
    @patch("src.save_docs_to_db.file_hash")
    @patch("src.save_docs_to_db.load_registry")
    @patch("src.save_docs_to_db._save_file_as_embeddings")
    def test_skips_already_indexed_files(
        self,
        mock_save_file: MagicMock,
        mock_load_reg: MagicMock,
        mock_hash: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Checks that files with a matching hash in the registry are skipped."""
        pdf = tmp_path / "old.pdf"
        pdf.write_bytes(b"%PDF-fake")

        with (
            patch.object(config.paths, "documents_folder", tmp_path),
            patch.object(config.paths, "supported_extensions", {".pdf"}),
        ):
            mock_load_reg.return_value = {"old.pdf": "same_hash"}
            mock_hash.return_value = "same_hash"

            save_all_files()

        mock_save_file.assert_not_called()

    @patch("src.save_docs_to_db.save_registry")
    @patch("src.save_docs_to_db.file_hash")
    @patch("src.save_docs_to_db.load_registry")
    @patch("src.save_docs_to_db._save_file_as_embeddings")
    def test_reindexes_modified_files(
        self,
        mock_save_file: MagicMock,
        mock_load_reg: MagicMock,
        mock_hash: MagicMock,
        mock_save_reg: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Checks that files with a changed hash are re-indexed."""
        pdf = tmp_path / "changed.pdf"
        pdf.write_bytes(b"%PDF-fake-v2")

        with (
            patch.object(config.paths, "documents_folder", tmp_path),
            patch.object(config.paths, "supported_extensions", {".pdf"}),
        ):
            mock_load_reg.return_value = {"changed.pdf": "old_hash"}
            mock_hash.return_value = "new_hash"
            mock_save_file.return_value = 3

            save_all_files()

        mock_save_file.assert_called_once_with(pdf)
        saved_reg = mock_save_reg.call_args[0][0]
        assert saved_reg["changed.pdf"] == "new_hash"

    @patch("src.save_docs_to_db.save_registry")
    @patch("src.save_docs_to_db.load_registry")
    @patch("src.save_docs_to_db._save_file_as_embeddings")
    def test_ignores_non_supported_extensions(
        self,
        mock_save_file: MagicMock,
        mock_load_reg: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Checks that files with unsupported extensions are ignored."""
        txt = tmp_path / "notes.txt"
        txt.write_text("not a pdf")

        with (
            patch.object(config.paths, "documents_folder", tmp_path),
            patch.object(config.paths, "supported_extensions", {".pdf"}),
        ):
            mock_load_reg.return_value = {}

            save_all_files()

        mock_save_file.assert_not_called()

    @patch("src.save_docs_to_db.save_registry")
    @patch("src.save_docs_to_db.load_registry")
    @patch("src.save_docs_to_db._save_file_as_embeddings")
    def test_empty_folder(
        self,
        mock_save_file: MagicMock,
        mock_load_reg: MagicMock,
        mock_save_reg: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Checks that an empty documents folder results in no indexing."""
        with (
            patch.object(config.paths, "documents_folder", tmp_path),
            patch.object(config.paths, "supported_extensions", {".pdf"}),
        ):
            mock_load_reg.return_value = {}

            save_all_files()

        mock_save_file.assert_not_called()
        mock_save_reg.assert_called_once_with({})
