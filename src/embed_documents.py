"""
Script to embed the documents and save them into the vector db.
"""

import uuid
from pathlib import Path

import fitz
from google import genai
from google.genai import types

from src.config import config
from src.utils import file_hash, get_index_vector_db, load_registry, save_registry


GEN_AI_CLIENT = genai.Client(api_key=config.chat_model.api_key)
PINECONE_IDX = get_index_vector_db()


def _embed_text(text: str) -> list[float]:
    """
    Embeds the text with the embedding model.

    Args:
        text: Text to embed.

    Returns:
        Embedding of the text.
    """

    response = GEN_AI_CLIENT.models.embed_content(
        model=config.embedding_model.embedding_model,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )

    return response.embeddings[0].values  # type: ignore


def _obtain_chunks(path: Path) -> list[dict[str, str | int]]:
    """
    Obtains the chunks from one file.

    Args:
        path: Path where the file is saved.

    Returns:

    """

    doc = fitz.open(str(path))
    chunks = []
    total_pages = len(doc)

    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if not text:
            print(f"Could not extract text from page {page_num} of file {path}.")
            continue
        chunks.append(
            {
                "text": text,
                "source": path.name,
                "page": page_num + 1,
                "total_pages": total_pages,
                "doc_type": "pdf",
            }
        )

    doc.close()

    return chunks


def _save_file(path: Path) -> int:
    """
    Saves a file into the vector db as embeddings.

    Args:
        path: Path where the file is saved.

    Returns:
        Number of extracted chunks.
    """

    chunks = _obtain_chunks(path)

    vectors: list[dict[str, str | list[float] | dict[str, int | str]]] = []
    for c in chunks:
        vec_id = str(uuid.uuid4())
        vectors.append(
            {
                "id": vec_id,
                "values": _embed_text(c["text"]),  # type: ignore
                "metadata": {
                    "text": c["text"],
                    "source": c["source"],
                    "page": c["page"],
                    "total_pages": c["total_pages"],
                    "image_file": c["image_file"],
                    "doc_type": c["doc_type"],
                },
            }
        )

    for i in range(0, len(vectors), 100):
        PINECONE_IDX.upsert(vectors=vectors[i : i + 100])  # type: ignore

    return len(chunks)


def save_all_files() -> None:
    """
    Saves all files that are present in the corresponding folder into the vector db as
    embeddings.
    """

    already_included_files = load_registry()
    all_files = [
        p
        for p in config.paths.documents_folder.iterdir()
        if p.suffix.lower() in config.paths.supported_extensions
    ]

    for path in all_files:
        h = file_hash(path)

        if already_included_files.get(path.name) == h:
            print(f"Skipping {path} as it's already saved.")
            continue

        print(f"Indexing {path}...")
        n = _save_file(path)
        already_included_files[path.name] = h
        print(f"Finished. Number of chunks: {n}.")

    save_registry(already_included_files)
