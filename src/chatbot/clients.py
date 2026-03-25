"""
Chat client implementations for different LLM providers. This is not done with Langchain
because it's not too complex and this way we avoid the dependencies.
"""

from typing import Literal, Protocol

import anthropic
import openai
from google import genai
from google.genai import types


class ChatClient(Protocol):
    """
    Protocol for chat clients.
    """

    def __init__(self, api_key: str) -> None: ...

    def chat(
        self,
        model: str,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Sends a chat request to the provider.

        Args:
            model: Model name.
            system_prompt: System prompt.
            messages: Conversation turns with 'role' and 'content' keys.

        Returns:
            Model response text.
        """


class GeminiChat(ChatClient):
    """Chat client for Gemini."""

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
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Sends a chat request to Gemini.

        Args:
            model: Model name.
            system_prompt: System prompt.
            messages: Conversation turns with "role" and "content" keys.

        Returns:
            Model response text.
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
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )

        if not isinstance(response.text, str):
            raise RuntimeError("Gemini could not generate a response.")

        return response.text


class OpenAIChat(ChatClient):
    """
    Chat client for OpenAI.
    """

    def __init__(self, api_key: str) -> None:
        """
        Constructor of the class.
        Args:
            api_key: OpenAI API key.
        """

        self._client = openai.OpenAI(api_key=api_key)

    def chat(
        self,
        model: str,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Sends a chat request to OpenAI.

        Args:
            model: Model name.
            system_prompt: System prompt.
            messages: Conversation turns with 'role' and 'content' keys.

        Returns:
            Model response text.
        """

        oai_messages: list[openai.types.chat.ChatCompletionMessageParam] = [
            openai.types.chat.ChatCompletionSystemMessageParam(
                role="system", content=system_prompt
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
            raise RuntimeError("Gemini could not generate a response.")

        return response.choices[0].message.content


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
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Sends a chat request to Anthropic.

        Args:
            model: Model name.
            system_prompt: System prompt.
            messages: Conversation turns with "role" and "content" keys.

        Returns:
            Model response text.
        """

        ant_messages: list[anthropic.types.MessageParam] = []
        for msg in messages:
            role: Literal["user", "assistant"] = (
                "user" if msg["role"] == "user" else "assistant"
            )
            ant_messages.append({"role": role, "content": msg["content"]})
        response = self._client.messages.create(
            model=model, system=system_prompt, messages=ant_messages, max_tokens=4096
        )

        return response.content[0].text  # type: ignore


class GroqChat(ChatClient):
    """
    Chat client for Groq.
    """

    def __init__(self, api_key: str) -> None:
        """
        Constructor of the class.

        Args:
            api_key: Groq API key.
        """

        self._client = openai.OpenAI(
            api_key=api_key, base_url="https://api.groq.com/openai/v1"
        )

    def chat(
        self,
        model: str,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Sends a chat request to Groq.

        Args:
            model: Model name.
            system_prompt: System prompt.
            messages: Conversation turns with "role" and "content" keys.

        Returns:
            Model response text.
        """

        oai_messages: list[openai.types.chat.ChatCompletionMessageParam] = [
            openai.types.chat.ChatCompletionSystemMessageParam(
                role="system", content=system_prompt
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
            raise RuntimeError("Groq could not generate a response.")

        return response.choices[0].message.content


CHAT_CLIENTS: dict[str, type[ChatClient]] = {
    "Gemini": GeminiChat,
    "OpenAI": OpenAIChat,
    "Anthropic": AnthropicChat,
    "Groq": GroqChat,
}


def create_chat_client(provider: str, api_key: str) -> ChatClient:
    """
    Creates a chat client for the given provider.

    Args:
        provider: Provider name (Gemini, OpenAI, Anthropic).
        api_key: API key for the provider.

    Returns:
        Chat client instance.
    """
    cls = CHAT_CLIENTS.get(provider)
    if cls is None:
        raise ValueError(f"Unknown provider: {provider}")
    return cls(api_key=api_key)
