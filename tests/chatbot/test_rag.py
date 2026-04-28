"""
Tests for src/chatbot/rag.py
"""

from unittest.mock import MagicMock

from src.chatbot.rag import RAGTool
from src.config import config


# Helpers


def _make_match(
    *,
    text: str,
    source: str,
    location: int,
    total_locations: int,
    doc_type: str,
    score: float,
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


def _make_rag_tool(
    embedding_values: list[float] | None = None,
    matches: list[MagicMock] | None = None,
    top_k: int = 3,
) -> tuple[RAGTool, MagicMock, MagicMock]:
    """Creates a RAGTool with mocked dependencies."""
    embedding_client = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = embedding_values or [0.1, 0.2]
    mock_response = MagicMock()
    mock_response.embeddings = [mock_embedding]
    embedding_client.models.embed_content.return_value = mock_response

    pinecone_index = MagicMock()
    pinecone_index.query.return_value = MagicMock(matches=matches or [])

    rag = RAGTool(embedding_client, pinecone_index, top_k=top_k)
    return rag, embedding_client, pinecone_index


# Tests


class TestEmbedQuery:
    """Tests for the _embed_query method."""

    def test_returns_embedding_values(self) -> None:
        """Checks that the embedding values from the API response are returned."""
        rag, *_ = _make_rag_tool(embedding_values=[0.1, 0.2])

        result = rag._embed_query("test question")

        assert result == [0.1, 0.2]

    def test_passes_correct_model_and_config(self) -> None:
        """Checks that the correct model name and contents are sent to the API."""
        rag, embedding_client, _ = _make_rag_tool()

        rag._embed_query("hello")

        call_kwargs = embedding_client.models.embed_content.call_args
        assert call_kwargs.kwargs["model"] == config.embedding_model.embedding_model
        assert call_kwargs.kwargs["contents"] == "hello"

    def test_raises_on_empty_response(self) -> None:
        """Checks that a RuntimeError is raised when the embedding response is empty."""
        rag, embedding_client, _ = _make_rag_tool()
        mock_response = MagicMock()
        mock_response.embeddings = []
        embedding_client.models.embed_content.return_value = mock_response

        try:
            rag._embed_query("test")
            assert False, "Expected RuntimeError"
        except RuntimeError:
            pass


class TestRetrieve:
    """Tests for the retrieve method."""

    def test_returns_formatted_chunks(self) -> None:
        """Checks that a Pinecone match is correctly formatted into a chunk dict."""
        match = _make_match(
            text="some text",
            source="doc.pdf",
            location=1,
            total_locations=10,
            doc_type="pdf",
            score=0.95123,
        )
        rag, *_ = _make_rag_tool(matches=[match])

        result = rag.retrieve("question")

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
        rag, _, pinecone_index = _make_rag_tool(top_k=7)

        rag.retrieve("q")

        call_kwargs = pinecone_index.query.call_args.kwargs
        assert call_kwargs["top_k"] == 7
        assert call_kwargs["include_metadata"] is True

    def test_multiple_matches(self) -> None:
        """Checks that multiple Pinecone matches are all returned in order."""
        matches = [
            _make_match(
                text="t1",
                source="a.pdf",
                location=1,
                total_locations=5,
                doc_type="pdf",
                score=0.9,
            ),
            _make_match(
                text="t2",
                source="b.pdf",
                location=3,
                total_locations=8,
                doc_type="pdf",
                score=0.7,
            ),
        ]
        rag, *_ = _make_rag_tool(matches=matches)

        result = rag.retrieve("q")

        assert len(result) == 2
        assert result[0]["source"] == "a.pdf"
        assert result[1]["source"] == "b.pdf"

    def test_empty_results(self) -> None:
        """Checks that an empty match list returns an empty list."""
        rag, *_ = _make_rag_tool()

        result = rag.retrieve("q")

        assert result == []


class TestTool:
    """Tests for the LangChain tool exposed by RAGTool."""

    def test_tool_has_correct_name(self) -> None:
        """Checks that the tool has the expected name."""
        rag, *_ = _make_rag_tool()
        assert rag.tool.name == "search_course_material"

    def test_tool_returns_json(self) -> None:
        """Checks that the tool returns a JSON string of chunks."""
        match = _make_match(
            text="text",
            source="doc.pdf",
            location=1,
            total_locations=5,
            doc_type="pdf",
            score=0.9,
        )
        rag, *_ = _make_rag_tool(matches=[match])

        result = rag.tool.invoke({"query": "test"})

        assert '"source": "doc.pdf"' in result

    def test_tool_returns_message_when_empty(self) -> None:
        """Checks that the tool returns a message when no chunks are found."""
        rag, *_ = _make_rag_tool()

        result = rag.tool.invoke({"query": "test"})

        assert result == "No relevant course material found."
