"""
Chat client factory and chat logic with optional RAG tool calling.
"""

import json
from typing import Callable

from groq import BadRequestError as GroqBadRequestError
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from openai import BadRequestError as OpenAIBadRequestError

from src.chatbot.rag import RAGTool
from src.config import config


CHAT_CLIENTS: dict[str, Callable[..., BaseChatModel]] = {
    "Gemini": ChatGoogleGenerativeAI,
    "OpenAI": ChatOpenAI,
    "Anthropic": ChatAnthropic,
    "Groq": ChatGroq,
}


def create_chat_client(provider: str, api_key: str) -> BaseChatModel:
    """
    Creates a LangChain chat model for the given provider.

    Args:
        provider: Provider name (Gemini, OpenAI, Anthropic, Groq).
        api_key: API key for the provider.

    Returns:
        LangChain chat model instance.

    Raises:
        ValueError: If the provider is not supported.
    """

    cls = CHAT_CLIENTS.get(provider)
    if cls is None:
        raise ValueError(f"Unknown provider: {provider}")

    model = config.chat_model.providers[provider]
    return cls(model=model, api_key=api_key)


def _safe_llm_response(
    llm_with_tools: Runnable, llm: BaseChatModel, messages: list[BaseMessage]
) -> AIMessage:
    """
    Invokes the LLM with tools, falling back to a plain call if the provider returns a
    malformed tool call.

    Args:
        llm_with_tools: LLM with tools bound.
        llm: Plain LLM without tools.
        messages: Conversation messages.

    Returns:
        The LLM response message.
    """

    try:
        return llm_with_tools.invoke(messages)
    except (OpenAIBadRequestError, GroqBadRequestError):
        return llm.invoke(messages)

    # if not isinstance(response, AIMessage):
    #    raise TypeError(f"Expected AIMessage, got {type(response).__name__}")


#
# return response


def chat(
    llm: BaseChatModel,
    question: str,
    rag_tool: RAGTool,
    chat_history: list[dict[str, str]] | None = None,
) -> tuple[str, list[dict[str, int | str | float]]]:
    """
    Sends a question to the LLM. The LLM may call the RAG tool if it needs course
    material, otherwise it answers directly.

    Args:
        llm: LangChain chat model.
        question: User question.
        rag_tool: RAGTool instance for course material retrieval.
        chat_history: Previous conversation turns with "role" and "content" keys.

    Returns:
        The answer string and a list of retrieved chunks (empty if RAG was not used).
    """

    llm_with_tools = llm.bind_tools([rag_tool.tool])

    # Reconstruct chat history
    messages: list[BaseMessage] = [
        SystemMessage(content=config.chat_model.system_prompt)
    ]
    for msg in chat_history or []:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=question))

    # Base response
    response = _safe_llm_response(llm_with_tools, llm, messages)
    messages.append(response)

    # Use tools (RAG) in case needed
    chunks: list[dict[str, int | str | float]] = []
    while response.tool_calls:
        for tool_call in response.tool_calls:
            result = rag_tool.tool.invoke(tool_call["args"])
            messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
            try:
                parsed = json.loads(result)
                if isinstance(parsed, list):
                    chunks.extend(parsed)
            except (json.JSONDecodeError, TypeError):
                pass
        response = _safe_llm_response(llm_with_tools, llm, messages)
        messages.append(response)

    return str(response.content), chunks
