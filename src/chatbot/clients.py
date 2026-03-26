"""
Chat client implementations for different LLM providers. This is not done with Langchain
because it's not too complex and this way we avoid the dependencies.
"""

from abc import ABC, abstractmethod
from typing import Callable, Literal

import anthropic
import openai
from google import genai
from google.genai import types

from src.config import config


class ChatClient(ABC):
    """
    Abstract base class for chat clients.
    """

    @abstractmethod
    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Sends a chat request to the provider.

        Args:
            model: Model name.
            messages: Conversation turns with 'role' and 'content' keys.

        Returns:
            Response of the model.
        """


class GeminiChat(ChatClient):
    """
    Chat client for Gemini.
    """

    def __init__(self, api_key: str) -> None:
        """
        Constructor of the class.

        Args:
            api_key: Gemini API key.
        """

        self._client = genai.Client(api_key=api_key)

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Sends a chat request to Gemini.

        Args:
            model: Model name.
            messages: Conversation turns with 'role' and 'content' keys.

        Returns:
            Response of the model.

        Raises:
            RuntimeError: If Gemini could not generate a response.
        """

        contents: list[types.Content] = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(
                types.Content(role=role, parts=[types.Part(text=msg["content"])])
            )
        response = self._client.models.generate_content(
            model=model,
            contents=contents,  # type: ignore
            config=types.GenerateContentConfig(
                system_instruction=config.chat_model.system_prompt
            ),
        )

        if not isinstance(response.text, str):
            raise RuntimeError("Gemini could not generate a response.")

        return response.text


class _OpenAICompatibleChat(ChatClient):
    """
    Shared base for providers that expose an OpenAI-compatible API.
    """

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        """
        Constructor of the class.

        Args:
            api_key: API key.
            base_url: Optional custom base URL.
        """

        if base_url:
            self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        else:
            self._client = openai.OpenAI(api_key=api_key)

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Sends a chat request to an OpenAI-compatible endpoint.

        Args:
            model: Model name.
            messages: Conversation turns with 'role' and 'content' keys.

        Returns:
            Response of the model.

        Raises:
            RuntimeError: If the provider could not generate a response.
        """

        oai_messages: list[openai.types.chat.ChatCompletionMessageParam] = [
            openai.types.chat.ChatCompletionSystemMessageParam(
                role="system", content=config.chat_model.system_prompt
            )
        ]
        for msg in messages:
            if msg["role"] == "user":
                oai_messages.append(
                    openai.types.chat.ChatCompletionUserMessageParam(
                        role="user", content=msg["content"]
                    )
                )
            else:
                oai_messages.append(
                    openai.types.chat.ChatCompletionAssistantMessageParam(
                        role="assistant", content=msg["content"]
                    )
                )
        response = self._client.chat.completions.create(
            model=model, messages=oai_messages
        )

        if not isinstance(response.choices[0].message.content, str):
            raise RuntimeError(
                f"{self.__class__.__name__} could not generate a response."
            )

        return response.choices[0].message.content


class OpenAIChat(_OpenAICompatibleChat):
    """Chat client for OpenAI."""

    def __init__(self, api_key: str) -> None:
        super().__init__(api_key)


class GroqChat(_OpenAICompatibleChat):
    """Chat client for Groq."""

    def __init__(self, api_key: str) -> None:
        super().__init__(api_key, base_url="https://api.groq.com/openai/v1")


class AnthropicChat(ChatClient):
    """
    Chat client for Anthropic.
    """

    def __init__(self, api_key: str) -> None:
        """
        Constructor of the class.

        Args:
            api_key: Anthropic API key.
        """

        self._client = anthropic.Anthropic(api_key=api_key)

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Sends a chat request to Anthropic.

        Args:
            model: Model name.
            messages: Conversation turns with 'role' and 'content' keys.

        Returns:
            Model response text.

        Raises:
            RuntimeError: If Anthropic could not generate a response.
        """

        ant_messages: list[anthropic.types.MessageParam] = []
        for msg in messages:
            role: Literal["user", "assistant"] = (
                "user" if msg["role"] == "user" else "assistant"
            )
            ant_messages.append({"role": role, "content": msg["content"]})
        response = self._client.messages.create(
            model=model,
            system=config.chat_model.system_prompt,
            messages=ant_messages,
            max_tokens=4_096,
        )

        block = response.content[0]
        if not hasattr(block, "text"):
            raise RuntimeError("Anthropic could not generate a response.")

        return block.text


CHAT_CLIENTS: dict[str, Callable[[str], ChatClient]] = {
    "Gemini": GeminiChat,
    "OpenAI": OpenAIChat,
    "Anthropic": AnthropicChat,
    "Groq": GroqChat,
}
# The Callable[[str], ChatClient] type means a function that takes a string (the API
# key) and returns a ChatClient instance.


def create_chat_client(provider: str, api_key: str) -> ChatClient:
    """
    Creates a chat client for the given provider.

    Args:
        provider: Provider name (Gemini, OpenAI, Anthropic).
        api_key: API key for the provider.

    Returns:
        Chat client instance.

    Raises:
        ValueError: If the provider is not supported.
    """

    cls = CHAT_CLIENTS.get(provider)

    if cls is None:
        raise ValueError(f"Unknown provider: {provider}")

    return cls(api_key)
