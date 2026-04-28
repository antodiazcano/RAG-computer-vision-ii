"""
Tests for src/chatbot/clients.py
"""

import json
from unittest.mock import MagicMock

import pytest
from groq import BadRequestError as GroqBadRequestError
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from openai import BadRequestError as OpenAIBadRequestError

from src.chatbot.clients import _safe_llm_response, chat, create_chat_client


class TestCreateChatClient:
    """Tests for the create_chat_client function."""

    def test_creates_gemini_client(self) -> None:
        """Checks that a ChatGoogleGenerativeAI instance is created for Gemini."""
        client = create_chat_client("Gemini", "fake-key")
        assert isinstance(client, ChatGoogleGenerativeAI)

    def test_creates_openai_client(self) -> None:
        """Checks that a ChatOpenAI instance is created for OpenAI."""
        client = create_chat_client("OpenAI", "fake-key")
        assert isinstance(client, ChatOpenAI)

    def test_creates_anthropic_client(self) -> None:
        """Checks that a ChatAnthropic instance is created for Anthropic."""
        client = create_chat_client("Anthropic", "fake-key")
        assert isinstance(client, ChatAnthropic)

    def test_creates_groq_client(self) -> None:
        """Checks that a ChatGroq instance is created for Groq."""
        client = create_chat_client("Groq", "fake-key")
        assert isinstance(client, ChatGroq)

    def test_raises_on_unknown_provider(self) -> None:
        """Checks that an unknown provider raises a ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_chat_client("Unknown", "fake-key")


class TestChat:
    """Tests for the chat function."""

    @staticmethod
    def _mock_rag_tool() -> MagicMock:
        """Creates a mock RAGTool."""
        rag_tool = MagicMock()
        rag_tool.tool = MagicMock()
        return rag_tool

    def test_direct_answer_without_tool(self) -> None:
        """Checks that a direct answer is returned when no tool is called."""
        llm = MagicMock()
        response = AIMessage(content="Hello!")
        response.tool_calls = []
        llm.bind_tools.return_value = llm
        llm.invoke.return_value = response

        answer, chunks = chat(llm, "Hi there", rag_tool=self._mock_rag_tool())

        assert answer == "Hello!"
        assert not chunks

    def test_answer_with_tool_call(self) -> None:
        """Checks that the RAG tool is called and chunks are returned."""
        llm = MagicMock()
        llm.bind_tools.return_value = llm

        tool_chunks = [
            {
                "text": "t",
                "source": "a.pdf",
                "location": 1,
                "total_locations": 5,
                "doc_type": "pdf",
                "score": 0.9,
            }
        ]

        tool_response = AIMessage(content="")
        tool_response.tool_calls = [
            {"id": "1", "name": "search_course_material", "args": {"query": "q"}}
        ]
        final_response = AIMessage(content="Based on the material...")
        final_response.tool_calls = []

        llm.invoke.side_effect = [tool_response, final_response]

        rag_tool = self._mock_rag_tool()
        rag_tool.tool.invoke.return_value = json.dumps(tool_chunks)

        answer, chunks = chat(llm, "What is a saliency map?", rag_tool=rag_tool)

        assert answer == "Based on the material..."
        assert len(chunks) == 1
        assert chunks[0]["source"] == "a.pdf"

    def test_empty_history(self) -> None:
        """Checks that chat works with no history."""
        llm = MagicMock()
        response = AIMessage(content="answer")
        response.tool_calls = []
        llm.bind_tools.return_value = llm
        llm.invoke.return_value = response

        answer, chunks = chat(llm, "q", rag_tool=self._mock_rag_tool(), chat_history=[])

        assert answer == "answer"
        assert not chunks

    def test_passes_chat_history(self) -> None:
        """Checks that chat history is included in the messages."""
        llm = MagicMock()
        response = AIMessage(content="answer")
        response.tool_calls = []
        llm.bind_tools.return_value = llm
        llm.invoke.return_value = response

        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        chat(llm, "q", rag_tool=self._mock_rag_tool(), chat_history=history)

        # The messages list is mutated after invoke, so check content not length
        messages = llm.invoke.call_args[0][0]
        contents = [m.content for m in messages]
        assert "hi" in contents
        assert "hello" in contents
        assert messages[1].content == "hi"
        assert messages[2].content == "hello"


class TestSafeLlmResponse:
    """Tests for the _safe_llm_response function."""

    def test_returns_response_from_llm_with_tools(self) -> None:
        """Checks that the response from llm_with_tools is returned on success."""
        expected = AIMessage(content="tool response")
        llm_with_tools = MagicMock()
        llm_with_tools.invoke.return_value = expected
        llm = MagicMock()

        result = _safe_llm_response(llm_with_tools, llm, [HumanMessage(content="q")])

        assert result == expected
        llm.invoke.assert_not_called()

    def test_falls_back_on_openai_bad_request(self) -> None:
        """Checks fallback to plain LLM on OpenAI BadRequestError."""
        fallback = AIMessage(content="fallback")
        llm_with_tools = MagicMock()
        llm_with_tools.invoke.side_effect = OpenAIBadRequestError(
            message="tool_use_failed", response=MagicMock(), body=None
        )
        llm = MagicMock()
        llm.invoke.return_value = fallback

        result = _safe_llm_response(llm_with_tools, llm, [HumanMessage(content="q")])

        assert result == fallback

    def test_falls_back_on_groq_bad_request(self) -> None:
        """Checks fallback to plain LLM on Groq BadRequestError."""
        fallback = AIMessage(content="fallback")
        llm_with_tools = MagicMock()
        llm_with_tools.invoke.side_effect = GroqBadRequestError(
            message="tool_use_failed", response=MagicMock(), body=None
        )
        llm = MagicMock()
        llm.invoke.return_value = fallback

        result = _safe_llm_response(llm_with_tools, llm, [HumanMessage(content="q")])

        assert result == fallback

    def test_does_not_catch_other_exceptions(self) -> None:
        """Checks that unrelated exceptions are not caught."""
        llm_with_tools = MagicMock()
        llm_with_tools.invoke.side_effect = RuntimeError("unexpected")
        llm = MagicMock()

        with pytest.raises(RuntimeError, match="unexpected"):
            _safe_llm_response(llm_with_tools, llm, [HumanMessage(content="q")])
