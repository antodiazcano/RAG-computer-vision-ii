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
    render_dpi = 150


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

    api_key = os.getenv("GEMINI_API_KEY")
    chat_model = "gemini-2.5-flash"
    system_prompt = (
        "You are a precise knowledge-base assistant. Answer using the context below.\n"
        "Always cite the source document and page number.\n"
        "If the answer is not in the context, answer the user, but warn that the RAG "
        "retrieval was not used."
    )


@dataclass
class PathsConfig:
    """
    Class to define the configuration of the paths of the project.
    """

    documents_folder = Path("data/documents")
    supported_extensions = {".pdf"}
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
