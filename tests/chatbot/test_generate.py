"""
Tests for src/chatbot/generate.py
"""

from unittest.mock import MagicMock, patch

from src.config import config


# Mock the module-level clients before importing generate
with (
    patch("src.utils.get_gen_ai_client", return_value=MagicMock()),
    patch("src.utils.get_index_vector_db", return_value=MagicMock()),
):
    from src.chatbot.generate import (
        EMBEDDING_CLIENT,
        PINECONE_IDX,
        _embed_query,
        _retrieve_from_vector_db,
        generate_answer,
    )


# Helpers


def _make_match(
    text: str, source: str, page: int, total_pages: int, doc_type: str, score: float
) -> MagicMock:
    """Creates a mock Pinecone match object with the given metadata and score."""
    m = MagicMock()
    m.metadata = {
        "text": text,
        "source": source,
        "page": page,
        "total_pages": total_pages,
        "doc_type": doc_type,
    }
    m.score = score
    return m


def _setup_embedding_mock() -> None:
    """Sets up the EMBEDDING_CLIENT mock to return a default embedding."""
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1, 0.2]
    mock_response = MagicMock()
    mock_response.embeddings = [mock_embedding]
    EMBEDDING_CLIENT.models.embed_content.return_value = mock_response


class TestEmbedQuery:
    """Tests for the _embed_query function."""

    def test_returns_embedding_values(self) -> None:
        """Checks that the embedding values from the API response are returned."""
        _setup_embedding_mock()

        result = _embed_query("test question")

        assert result == [0.1, 0.2]

    def test_passes_correct_model_and_config(self) -> None:
        """Checks that the correct model name and contents are sent to the API."""
        _setup_embedding_mock()

        _embed_query("hello")

        call_kwargs = EMBEDDING_CLIENT.models.embed_content.call_args
        assert call_kwargs.kwargs["model"] == config.embedding_model.embedding_model
        assert call_kwargs.kwargs["contents"] == "hello"


class TestRetrieveFromVectorDb:
    """Tests for the _retrieve_from_vector_db function."""

    def setup_method(self) -> None:
        """Resets mocks and sets up a default embedding response."""
        EMBEDDING_CLIENT.models.embed_content.reset_mock()
        PINECONE_IDX.query.reset_mock()
        _setup_embedding_mock()

    def test_returns_formatted_chunks(self) -> None:
        """Checks that a Pinecone match is correctly formatted into a chunk dict."""
        match = _make_match("some text", "doc.pdf", 1, 10, "pdf", 0.95123)
        PINECONE_IDX.query.return_value = MagicMock(matches=[match])

        result = _retrieve_from_vector_db("question")

        assert len(result) == 1
        assert result[0] == {
            "text": "some text",
            "source": "doc.pdf",
            "page": 1,
            "total_pages": 10,
            "doc_type": "pdf",
            "score": 0.9512,
        }

    def test_respects_top_k(self) -> None:
        """Checks that the top_k parameter is forwarded to the Pinecone query."""
        PINECONE_IDX.query.return_value = MagicMock(matches=[])

        _retrieve_from_vector_db("q", top_k=7)

        call_kwargs = PINECONE_IDX.query.call_args.kwargs
        assert call_kwargs["top_k"] == 7
        assert call_kwargs["include_metadata"] is True

    def test_multiple_matches(self) -> None:
        """Checks that multiple Pinecone matches are all returned in order."""
        matches = [
            _make_match("t1", "a.pdf", 1, 5, "pdf", 0.9),
            _make_match("t2", "b.pdf", 3, 8, "pdf", 0.7),
        ]
        PINECONE_IDX.query.return_value = MagicMock(matches=matches)

        result = _retrieve_from_vector_db("q")

        assert len(result) == 2
        assert result[0]["source"] == "a.pdf"
        assert result[1]["source"] == "b.pdf"

    def test_empty_results(self) -> None:
        """Checks that an empty match list returns an empty list."""
        PINECONE_IDX.query.return_value = MagicMock(matches=[])

        result = _retrieve_from_vector_db("q")

        assert result == []


class TestGenerateAnswer:
    """Tests for the generate_answer function."""

    def setup_method(self) -> None:
        """Resets mocks and sets up a default embedding response."""
        EMBEDDING_CLIENT.models.embed_content.reset_mock()
        PINECONE_IDX.query.reset_mock()
        _setup_embedding_mock()

    def test_returns_answer_and_chunks(self) -> None:
        """Checks that the function returns a string answer and the retrieved chunks."""
        match = _make_match("text", "doc.pdf", 1, 5, "pdf", 0.9)
        PINECONE_IDX.query.return_value = MagicMock(matches=[match])
        mock_chat = MagicMock()
        mock_chat.chat.return_value = "answer"

        answer, chunks = generate_answer("question", mock_chat, "Gemini")

        assert answer == "answer"
        assert len(chunks) == 1
        assert chunks[0]["source"] == "doc.pdf"

    def test_passes_top_k(self) -> None:
        """Checks that the top_k parameter is forwarded to the vector db query."""
        PINECONE_IDX.query.return_value = MagicMock(matches=[])
        mock_chat = MagicMock()
        mock_chat.chat.return_value = "ok"

        generate_answer("q", mock_chat, "Gemini", top_k=5)

        assert PINECONE_IDX.query.call_args.kwargs["top_k"] == 5

    def test_empty_retrieval(self) -> None:
        """Checks that an empty retrieval still returns a valid answer string."""
        PINECONE_IDX.query.return_value = MagicMock(matches=[])
        mock_chat = MagicMock()
        mock_chat.chat.return_value = "ok"

        answer, chunks = generate_answer("q", mock_chat, "Gemini")

        assert chunks == []
        assert isinstance(answer, str)

    def test_passes_chat_history(self) -> None:
        """Checks that chat_history is included in the messages sent to the client."""
        PINECONE_IDX.query.return_value = MagicMock(matches=[])
        mock_chat = MagicMock()
        mock_chat.chat.return_value = "ok"
        history = [{"role": "user", "content": "hi"}]

        generate_answer("q", mock_chat, "Gemini", chat_history=history)

        messages = mock_chat.chat.call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["content"] == "hi"

    def test_uses_correct_model_for_provider(self) -> None:
        """Checks that the correct model is used based on the provider."""
        PINECONE_IDX.query.return_value = MagicMock(matches=[])
        mock_chat = MagicMock()
        mock_chat.chat.return_value = "ok"

        generate_answer("q", mock_chat, "OpenAI")

        model = mock_chat.chat.call_args.kwargs["model"]
        assert model == config.chat_model.providers["OpenAI"]
