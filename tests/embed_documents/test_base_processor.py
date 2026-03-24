"""
Tests for src/embed_documents/base_processor.py
"""

from pathlib import Path
from unittest.mock import MagicMock, patch


from src.config import config


# Mock the module-level clients before importing
with (
    patch("src.utils.get_gen_ai_client", return_value=MagicMock()),
    patch("src.utils.get_index_vector_db", return_value=MagicMock()),
):
    from src.embed_documents.base_processor import (
        GEN_AI_CLIENT,
        PINECONE_IDX,
        Processor,
    )


class StubProcessor(Processor):
    """Concrete subclass of Processor for testing, with controllable chunks."""

    def __init__(
        self, path: Path, chunks: list[dict[str, str | int]], batch_size: int = 50
    ) -> None:
        """Initializes the stub with a predefined list of chunks."""
        super().__init__(path, batch_size)
        self._chunks = chunks

    def _obtain_chunks(self) -> list[dict[str, str | int]]:
        """Returns the predefined chunks."""
        return self._chunks


def _make_chunks(n: int, source: str = "doc.pdf") -> list[dict[str, str | int]]:
    """Creates n dummy chunks for testing.

    Args:
        n: Number of chunks to create.
        source: Source filename.

    Returns:
        List of chunk dicts.
    """
    return [
        {
            "text": f"text {i}",
            "source": source,
            "page": i,
            "total_pages": n,
            "doc_type": "pdf",
        }
        for i in range(1, n + 1)
    ]


def _setup_embed_mock(values: list[float] | None = None) -> None:
    """Sets up the GEN_AI_CLIENT embed mock.

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


class TestInit:
    """Tests for the Processor constructor."""

    def test_stores_path_and_default_batch_size(self, tmp_path: Path) -> None:
        """Checks that path and default batch_size are stored correctly."""
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"fake")
        proc = StubProcessor(f, [])

        assert proc.path == f
        assert proc.batch_size == 50

    def test_custom_batch_size(self, tmp_path: Path) -> None:
        """Checks that a custom batch_size is stored correctly."""
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"fake")
        proc = StubProcessor(f, [], batch_size=10)

        assert proc.batch_size == 10


class TestEmbedText:
    """Tests for the _embed_text static method."""

    def test_returns_embedding_values(self) -> None:
        """Checks that the embedding values from the API response are returned."""
        _setup_embed_mock([0.5, 0.6])

        result = Processor._embed_text("some text")

        assert result == [0.5, 0.6]

    def test_passes_correct_model_and_task_type(self) -> None:
        """Checks that the correct model and RETRIEVAL_DOCUMENT task type are used."""
        _setup_embed_mock()

        Processor._embed_text("hello")

        call_kwargs = GEN_AI_CLIENT.models.embed_content.call_args
        assert call_kwargs.kwargs["model"] == config.embedding_model.embedding_model
        assert call_kwargs.kwargs["contents"] == "hello"


class TestProcess:
    """Tests for the process method."""

    def setup_method(self) -> None:
        """Resets mocks before each test."""
        GEN_AI_CLIENT.models.embed_content.reset_mock()
        PINECONE_IDX.upsert.reset_mock()
        _setup_embed_mock()

    @patch("src.embed_documents.base_processor.save_registry")
    @patch("src.embed_documents.base_processor.file_hash")
    @patch("src.embed_documents.base_processor.load_registry")
    def test_returns_chunk_count(
        self,
        mock_load_reg: MagicMock,
        mock_hash: MagicMock,
        mock_save_reg: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Checks that the number of processed chunks is returned."""
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"fake")
        mock_load_reg.return_value = {}
        mock_hash.return_value = "new_hash"

        proc = StubProcessor(f, _make_chunks(3))
        result = proc.process()

        assert result == 3

    @patch("src.embed_documents.base_processor.save_registry")
    @patch("src.embed_documents.base_processor.file_hash")
    @patch("src.embed_documents.base_processor.load_registry")
    def test_skips_already_indexed(
        self,
        mock_load_reg: MagicMock,
        mock_hash: MagicMock,
        mock_save_reg: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Checks that a file with matching hash in the registry is skipped."""
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"fake")
        mock_load_reg.return_value = {"doc.pdf": "same_hash"}
        mock_hash.return_value = "same_hash"

        proc = StubProcessor(f, _make_chunks(2))
        result = proc.process()

        assert result == 0
        PINECONE_IDX.upsert.assert_not_called()
        mock_save_reg.assert_not_called()

    @patch("src.embed_documents.base_processor.save_registry")
    @patch("src.embed_documents.base_processor.file_hash")
    @patch("src.embed_documents.base_processor.load_registry")
    def test_upserts_vectors_to_pinecone(
        self,
        mock_load_reg: MagicMock,
        mock_hash: MagicMock,
        mock_save_reg: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Checks that vectors are upserted with correct values and metadata."""
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"fake")
        mock_load_reg.return_value = {}
        mock_hash.return_value = "new_hash"

        proc = StubProcessor(f, _make_chunks(1))
        proc.process()

        PINECONE_IDX.upsert.assert_called_once()
        vectors = PINECONE_IDX.upsert.call_args.kwargs["vectors"]
        assert len(vectors) == 1
        assert vectors[0]["values"] == [0.1, 0.2]
        assert vectors[0]["metadata"]["text"] == "text 1"
        assert vectors[0]["metadata"]["source"] == "doc.pdf"

    @patch("src.embed_documents.base_processor.save_registry")
    @patch("src.embed_documents.base_processor.file_hash")
    @patch("src.embed_documents.base_processor.load_registry")
    def test_batches_large_uploads(
        self,
        mock_load_reg: MagicMock,
        mock_hash: MagicMock,
        mock_save_reg: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Checks that vectors are upserted in batches of the configured size."""
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"fake")
        mock_load_reg.return_value = {}
        mock_hash.return_value = "new_hash"

        proc = StubProcessor(f, _make_chunks(120), batch_size=50)
        proc.process()

        assert PINECONE_IDX.upsert.call_count == 3  # 50 + 50 + 20

    @patch("src.embed_documents.base_processor.save_registry")
    @patch("src.embed_documents.base_processor.file_hash")
    @patch("src.embed_documents.base_processor.load_registry")
    def test_updates_registry_after_indexing(
        self,
        mock_load_reg: MagicMock,
        mock_hash: MagicMock,
        mock_save_reg: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Checks that the registry is updated with the new file hash after indexing."""
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"fake")
        mock_load_reg.return_value = {}
        mock_hash.return_value = "new_hash"

        proc = StubProcessor(f, _make_chunks(1))
        proc.process()

        mock_save_reg.assert_called_once()
        saved_reg = mock_save_reg.call_args[0][0]
        assert saved_reg["doc.pdf"] == "new_hash"

    @patch("src.embed_documents.base_processor.save_registry")
    @patch("src.embed_documents.base_processor.file_hash")
    @patch("src.embed_documents.base_processor.load_registry")
    def test_no_chunks_still_updates_registry(
        self,
        mock_load_reg: MagicMock,
        mock_hash: MagicMock,
        mock_save_reg: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Checks that a file with no chunks still gets registered."""
        f = tmp_path / "empty.pdf"
        f.write_bytes(b"fake")
        mock_load_reg.return_value = {}
        mock_hash.return_value = "hash_empty"

        proc = StubProcessor(f, [])
        result = proc.process()

        assert result == 0
        PINECONE_IDX.upsert.assert_not_called()
        saved_reg = mock_save_reg.call_args[0][0]
        assert saved_reg["empty.pdf"] == "hash_empty"


class TestGetDocumentHash:
    """Tests for the get_document_hash method."""

    def test_returns_file_hash(self, tmp_path: Path) -> None:
        """Checks that get_document_hash returns the MD5 hash of the file."""
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"content")

        proc = StubProcessor(f, [])

        import hashlib

        expected = hashlib.md5(b"content").hexdigest()
        assert proc.get_document_hash() == expected
