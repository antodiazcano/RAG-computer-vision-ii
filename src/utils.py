"""
Script to write some utility functions.
"""

import hashlib
import json
from json import JSONDecodeError
from pathlib import Path

from google import genai
from google.genai.client import Client
from pinecone import Pinecone
from pinecone.db_data.index import Index

from src.config import config


def file_hash(path: Path) -> str:
    """
    Hashes the file of a path.

    Args:
        path: Path to hash.

    Returns:
        Hash of the path.
    """

    return hashlib.md5(path.read_bytes()).hexdigest()


def load_registry() -> dict[str, str]:
    """
    Loads the registry of the already indexed files.
    """

    try:
        return json.loads(config.paths.registry_path.read_text(encoding="utf-8"))
    except (JSONDecodeError, FileNotFoundError):
        print("No registry found. Starting with an empty registry.")
        return {}


def save_registry(reg: dict[str, str]) -> None:
    """
    Updates (saves again) the indexed files.
    """

    config.paths.registry_path.write_text(
        json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def get_gen_ai_client() -> Client:
    """
    Obtains the gen AI client.

    Returns:
        Gen AI client.
    """

    return genai.Client(api_key=config.embedding_model.api_key)


def get_index_vector_db() -> Index:
    """
    Obtains the index of the db.

    Returns:
        Index of the db.

    Raises:
        ValueError: If the index name of the db is not found in the environment.
    """

    pc = Pinecone(api_key=config.vector_db.pinecone_api_key)

    if not isinstance(config.vector_db.pinecone_index_name, str):
        raise ValueError("Not index name found in the environment!")

    return pc.Index(config.vector_db.pinecone_index_name)
