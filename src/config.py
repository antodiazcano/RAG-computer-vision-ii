"""
Configuration of the project.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass
class EmbeddingModelConfig:
    """
    Class to define the configuration of the embedding model.
    """

    api_key = os.getenv("GEMINI_API_KEY")
    embedding_model = "gemini-embedding-2-preview"
    embedding_dim = 768


@dataclass
class VectorDBConfig:
    """
    Class to define the configuration of the vector db.
    """

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    pinecone_cloud = os.getenv("PINECONE_CLOUD")
    pinecone_region = os.getenv("PINECONE_REGION")


@dataclass
class ChatModelConfig:
    """
    Class to define the configuration of the chat model.
    """

    groq_api_key = os.getenv("GROQ_API_KEY")
    providers = {
        "Gemini": "gemini-2.5-flash",
        "OpenAI": "gpt-4.1-nano",
        "Anthropic": "claude-sonnet-4-20250514",
        "Groq": "llama-3.3-70b-versatile",
    }
    system_prompt: str = (
        "You are a precise and helpful academic assistant.\n\n"
        "You may use two sources of knowledge:\n"
        "1) The provided context (retrieved from a vector database)\n"
        "2) Your general knowledge\n\n"
        "Follow these rules:\n"
        "- If the answer is clearly supported by the context, use it as the primary "
        "source.\n"
        "- If the context is partially relevant, combine it with your general "
        "knowledge.\n"
        "- If the context is irrelevant or does not contain the answer, ignore it.\n\n"
        "When using the context:\n"
        "- Cite the source document and page number.\n"
        "- Do not invent citations.\n\n"
        "When the context does NOT contain the answer:\n"
        "- Explicitly say that the answer was not found in the provided documents.\n"
        "- Then provide the best possible answer using your general knowledge.\n\n"
        "Always be clear, concise, and accurate."
    )


@dataclass
class PathsConfig:
    """
    Class to define the configuration of the paths of the project.
    """

    documents_folder = Path("data/documents")
    supported_extensions = {".pdf", ".tex"}
    registry_path = Path("data/indexed_registry.json")


@dataclass
class Config:
    """
    Main configuration class.
    """

    embedding_model = EmbeddingModelConfig()
    vector_db = VectorDBConfig()
    chat_model = ChatModelConfig()
    paths = PathsConfig()


config = Config()
