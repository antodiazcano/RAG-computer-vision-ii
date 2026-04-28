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
        "You are a precise and helpful academic assistant for a Computer Vision II "
        "course.\n\n"
        "You have access to a search tool that retrieves information from the course "
        "documents. Use it when the question is about course-specific topics, "
        "concepts, or formulas. For general questions (greetings, common knowledge, "
        "etc.), answer directly without using the tool.\n\n"
        "When you use the search tool:\n"
        "- Cite the source document and section/page.\n"
        "- Do not invent citations.\n\n"
        "When you answer without the tool:\n"
        "- Do not cite any sources.\n\n"
        "Always be clear, didactic, and accurate."
    )
    tool_rag_prompt = (
        "Search the Computer Vision II course documents for information relevant to "
        "the query. Use this tool when the user asks about course-specific topics, "
        "concepts, formulas, or anything that might be covered in the course material."
    )


@dataclass
class PathsConfig:
    """
    Class to define the configuration of the paths of the project.
    """

    documents_folder = Path("data/documents")
    supported_extensions = {".pdf", ".tex"}
    registry_path = Path("data/indexed_registry.json")
    corpus_index_path = Path("data/corpus_index.json")


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
