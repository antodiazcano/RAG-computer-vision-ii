"""
Tests for src/chatbot/generate.py
"""

from unittest.mock import MagicMock

from src.chatbot.generate import RAGEngine
from src.config import config


# Helpers


def _make_match(
    text: str, source: str, location: int, total_locations: int, doc_type: str, score: float
) -> MagicMock:
    """Creates a mock Pinecone match object with the given metadata and score."""
    m = MagicMock()
    m.metadata = {
        "text": text,
        "source": source,
        "location": location,
        "total_locations": total_locations,
        "doc_type": doc_type,
    }
    m.score = score
    return m


def _make_engine(
    embedding_values: list[float] | None = None,
    matches: list[MagicMock] | None = None,
    chat_response: str = "answer",
) -> tuple[RAGEngine, MagicMock, MagicMock, MagicMock]:
    """Creates a RAGEngine with mocked dependencies and returns it along with the mocks."""
    embedding_client = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = embedding_values or [0.1, 0.2]
    mock_response = MagicMock()
    mock_response.embeddings = [mock_embedding]
    embedding_client.models.embed_content.return_value = mock_response

    pinecone_index = MagicMock()
    pinecone_index.query.return_value = MagicMock(matches=matches or [])

    chat_client = MagicMock()
    chat_client.chat.return_value = chat_response

    engine = RAGEngine(
        chat_client=chat_client,
        model="test-model",
        embedding_client=embedding_client,
        pinecone_index=pinecone_index,
    )
    return engine, embedding_client, pinecone_index, chat_client


class TestEmbedQuery:
    """Tests for the _embed_query method."""

    def test_returns_embedding_values(self) -> None:
        """Checks that the embedding values from the API response are returned."""
        engine, *_ = _make_engine(embedding_values=[0.1, 0.2])

        result = engine._embed_query("test question")

        assert result == [0.1, 0.2]

    def test_passes_correct_model_and_config(self) -> None:
        """Checks that the correct model name and contents are sent to the API."""
        engine, embedding_client, *_ = _make_engine()

        engine._embed_query("hello")

        call_kwargs = embedding_client.models.embed_content.call_args
        assert call_kwargs.kwargs["model"] == config.embedding_model.embedding_model
        assert call_kwargs.kwargs["contents"] == "hello"

    def test_raises_on_empty_response(self) -> None:
        """Checks that a RuntimeError is raised when the embedding response is empty."""
        engine, embedding_client, *_ = _make_engine()
        mock_response = MagicMock()
        mock_response.embeddings = []
        embedding_client.models.embed_content.return_value = mock_response

        try:
            engine._embed_query("test")
            assert False, "Expected RuntimeError"
        except RuntimeError:
            pass


class TestRetrieveFromVectorDb:
    """Tests for the _retrieve_from_vector_db method."""

    def test_returns_formatted_chunks(self) -> None:
        """Checks that a Pinecone match is correctly formatted into a chunk dict."""
        match = _make_match("some text", "doc.pdf", 1, 10, "pdf", 0.95123)
        engine, *_ = _make_engine(matches=[match])

        result = engine._retrieve_from_vector_db("question")

        assert len(result) == 1
        assert result[0] == {
            "text": "some text",
            "source": "doc.pdf",
            "location": 1,
            "total_locations": 10,
            "doc_type": "pdf",
            "score": 0.9512,
        }

    def test_respects_top_k(self) -> None:
        """Checks that the top_k parameter is forwarded to the Pinecone query."""
        engine, _, pinecone_index, _ = _make_engine()

        engine._retrieve_from_vector_db("q", top_k=7)

        call_kwargs = pinecone_index.query.call_args.kwargs
        assert call_kwargs["top_k"] == 7
        assert call_kwargs["include_metadata"] is True

    def test_multiple_matches(self) -> None:
        """Checks that multiple Pinecone matches are all returned in order."""
        matches = [
            _make_match("t1", "a.pdf", 1, 5, "pdf", 0.9),
            _make_match("t2", "b.pdf", 3, 8, "pdf", 0.7),
        ]
        engine, *_ = _make_engine(matches=matches)

        result = engine._retrieve_from_vector_db("q")

        assert len(result) == 2
        assert result[0]["source"] == "a.pdf"
        assert result[1]["source"] == "b.pdf"

    def test_empty_results(self) -> None:
        """Checks that an empty match list returns an empty list."""
        engine, *_ = _make_engine()

        result = engine._retrieve_from_vector_db("q")

        assert result == []


class TestGenerateAnswer:
    """Tests for the generate_answer method."""

    def test_returns_answer_and_chunks(self) -> None:
        """Checks that the method returns a string answer and the retrieved chunks."""
        match = _make_match("text", "doc.pdf", 1, 5, "pdf", 0.9)
        engine, *_ = _make_engine(matches=[match])

        answer, chunks = engine.generate_answer("question")

        assert answer == "answer"
        assert len(chunks) == 1
        assert chunks[0]["source"] == "doc.pdf"

    def test_passes_top_k(self) -> None:
        """Checks that the top_k parameter is forwarded to the vector db query."""
        engine, _, pinecone_index, _ = _make_engine()

        engine.generate_answer("q", top_k=5)

        assert pinecone_index.query.call_args.kwargs["top_k"] == 5

    def test_empty_retrieval(self) -> None:
        """Checks that an empty retrieval still returns a valid answer string."""
        engine, *_ = _make_engine()

        answer, chunks = engine.generate_answer("q")

        assert chunks == []
        assert isinstance(answer, str)

    def test_passes_chat_history(self) -> None:
        """Checks that chat_history is included in the messages sent to the client."""
        engine, _, _, chat_client = _make_engine()
        history = [{"role": "user", "content": "hi"}]

        engine.generate_answer("q", chat_history=history)

        messages = chat_client.chat.call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["content"] == "hi"

    def test_uses_model_from_constructor(self) -> None:
        """Checks that the model passed to the constructor is used for chat."""
        engine, _, _, chat_client = _make_engine()

        engine.generate_answer("q")

        model = chat_client.chat.call_args.kwargs["model"]
        assert model == "test-model"
