"""
Tests for src/utils.py
"""

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import config
from src.utils import (
    file_hash,
    get_gen_ai_client,
    get_index_vector_db,
    load_registry,
    save_registry,
)


class TestFileHash:
    """Tests for the file_hash function."""

    def test_returns_correct_md5(self, tmp_path: Path) -> None:
        """Checks that the returned hash matches the expected MD5 digest."""
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello world")
        expected = hashlib.md5(b"hello world").hexdigest()
        assert file_hash(f) == expected

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        """Checks that files with different content produce different hashes."""
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"aaa")
        f2.write_bytes(b"bbb")
        assert file_hash(f1) != file_hash(f2)

    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        """Checks that files with identical content produce the same hash."""
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"same")
        f2.write_bytes(b"same")
        assert file_hash(f1) == file_hash(f2)

    def test_empty_file(self, tmp_path: Path) -> None:
        """Checks that an empty file returns the MD5 of an empty byte string."""
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        expected = hashlib.md5(b"").hexdigest()
        assert file_hash(f) == expected


class TestLoadRegistry:
    """Tests for the load_registry function."""

    def test_loads_valid_json(self, tmp_path: Path) -> None:
        """Checks that a valid JSON registry file is loaded correctly."""
        registry = {"doc.pdf": "abc123"}
        reg_path = tmp_path / "registry.json"
        reg_path.write_text(json.dumps(registry), encoding="utf-8")

        with patch("src.utils.config.paths.registry_path", reg_path):
            assert load_registry() == registry

    def test_returns_empty_on_missing_file(self, tmp_path: Path) -> None:
        """Checks that a missing file returns an empty dict."""
        reg_path = tmp_path / "nonexistent.json"

        with patch("src.utils.config.paths.registry_path", reg_path):
            assert load_registry() == {}

    def test_returns_empty_on_empty_file(self, tmp_path: Path) -> None:
        """Checks that an empty file returns an empty dict."""
        reg_path = tmp_path / "registry.json"
        reg_path.write_text("", encoding="utf-8")

        with patch("src.utils.config.paths.registry_path", reg_path):
            assert load_registry() == {}

    def test_returns_empty_on_invalid_json(self, tmp_path: Path) -> None:
        """Checks that malformed JSON returns an empty dict."""
        reg_path = tmp_path / "registry.json"
        reg_path.write_text("{broken", encoding="utf-8")

        with patch("src.utils.config.paths.registry_path", reg_path):
            assert load_registry() == {}


class TestSaveRegistry:
    """Tests for the save_registry function."""

    def test_saves_and_reads_back(self, tmp_path: Path) -> None:
        """Checks that a saved registry can be read back correctly."""
        reg_path = tmp_path / "registry.json"
        registry = {"a.pdf": "hash1", "b.pdf": "hash2"}

        with patch("src.utils.config.paths.registry_path", reg_path):
            save_registry(registry)

        saved = json.loads(reg_path.read_text(encoding="utf-8"))
        assert saved == registry

    def test_saves_empty_registry(self, tmp_path: Path) -> None:
        """Checks that an empty registry is saved as an empty JSON object."""
        reg_path = tmp_path / "registry.json"

        with patch("src.utils.config.paths.registry_path", reg_path):
            save_registry({})

        saved = json.loads(reg_path.read_text(encoding="utf-8"))
        assert saved == {}

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        """Checks that saving overwrites the previous registry content."""
        reg_path = tmp_path / "registry.json"
        reg_path.write_text(json.dumps({"old": "data"}), encoding="utf-8")

        with patch("src.utils.config.paths.registry_path", reg_path):
            save_registry({"new": "data"})

        saved = json.loads(reg_path.read_text(encoding="utf-8"))
        assert saved == {"new": "data"}


class TestGetGenAiClient:
    """Tests for the get_gen_ai_client function."""

    @patch("src.utils.genai.Client")
    def test_creates_client_with_api_key(self, mock_client_cls: MagicMock) -> None:
        """Checks that the Gemini client is created with the configured API key."""
        mock_client_cls.return_value = MagicMock()
        client = get_gen_ai_client()
        mock_client_cls.assert_called_once_with(api_key=config.embedding_model.api_key)
        assert client == mock_client_cls.return_value


class TestGetIndexVectorDb:
    """Tests for the get_index_vector_db function."""

    @patch("src.utils.Pinecone")
    def test_returns_index(self, mock_pinecone_cls: MagicMock) -> None:
        """Checks that the Pinecone index is returned when the name is configured."""
        mock_pc = MagicMock()
        mock_pinecone_cls.return_value = mock_pc

        with patch("src.utils.config.vector_db.pinecone_index_name", "my-index"):
            idx = get_index_vector_db()

        mock_pinecone_cls.assert_called_once()
        mock_pc.Index.assert_called_once_with("my-index")
        assert idx == mock_pc.Index.return_value

    @patch("src.utils.Pinecone")
    def test_raises_when_no_index_name(self, mock_pinecone_cls: MagicMock) -> None:
        """Checks that a ValueError is raised when the index name is not set."""
        mock_pinecone_cls.return_value = MagicMock()

        with patch("src.utils.config.vector_db.pinecone_index_name", None):
            with pytest.raises(ValueError, match="Not index name found"):
                get_index_vector_db()
