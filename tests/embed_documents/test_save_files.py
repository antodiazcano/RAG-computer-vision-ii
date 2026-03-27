"""
Tests for src/embed_documents/save_files.py
"""

from unittest.mock import MagicMock

from src.embed_documents.save_files import delete_vectors_by_metadata


class TestDeleteVectorsByMetadata:
    """Tests for the delete_vectors_by_metadata function."""

    def test_calls_delete_with_filter(self) -> None:
        """Checks that the Pinecone index delete is called with the given filter."""
        index = MagicMock()
        metadata_filter = {"source": "file.pdf"}

        delete_vectors_by_metadata(index, metadata_filter)

        index.delete.assert_called_once_with(filter={"source": "file.pdf"})
