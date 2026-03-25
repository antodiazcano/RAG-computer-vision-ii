"""
Tests for src/chatbot/clients.py
"""

from unittest.mock import MagicMock

import pytest

from src.chatbot.clients import (
    AnthropicChat,
    GeminiChat,
    OpenAIChat,
    create_chat_client,
)


class TestCreateChatClient:
    """Tests for the create_chat_client function."""

    def test_creates_gemini_client(self) -> None:
        """Checks that a GeminiChat instance is created for the Gemini provider."""
        client = create_chat_client("Gemini", "fake-key")
        assert isinstance(client, GeminiChat)

    def test_creates_openai_client(self) -> None:
        """Checks that an OpenAIChat instance is created for the OpenAI provider."""
        client = create_chat_client("OpenAI", "fake-key")
        assert isinstance(client, OpenAIChat)

    def test_creates_anthropic_client(self) -> None:
        """Checks that an AnthropicChat instance is created for the Anthropic provider."""
        client = create_chat_client("Anthropic", "fake-key")
        assert isinstance(client, AnthropicChat)

    def test_raises_on_unknown_provider(self) -> None:
        """Checks that an unknown provider raises a ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_chat_client("Unknown", "fake-key")
