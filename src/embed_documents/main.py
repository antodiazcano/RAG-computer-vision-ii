"""
Script with the main functions to manage the vector db: deleting vectors and indexing
documents.
"""

from pathlib import Path
from typing import Any

from src.config import config
from src.embed_documents.pdf_processor import PDFProcessor
from src.embed_documents.tex_processor import TEXProcessor
from src.utils import get_index_vector_db, load_registry, save_registry


PROCESSORS = {
    ".pdf": PDFProcessor,
    ".tex": TEXProcessor,
}


def delete_vectors_by_metadata(
    metadata_filter: dict[str, Any], remove_from_registry: str | None = None
) -> None:
    """
    Deletes vectors from the vector db that match the given metadata filter.

    Args:
        metadata_filter: Pinecone metadata filter (e.g. {"source": "file.pdf"}).
        remove_from_registry: If provided, removes this filename from the registry.
    """

    idx = get_index_vector_db()
    idx.delete(filter=metadata_filter)
    print(f"Deleted vectors matching {metadata_filter}.")

    if remove_from_registry:
        reg = load_registry()
        if reg.pop(remove_from_registry, None):
            save_registry(reg)
            print(f"Removed '{remove_from_registry}' from registry.")


def save_all_files(folder: Path | None = None) -> dict[str, int]:
    """
    Indexes all supported files in a folder into the vector db.

    Args:
        folder: Folder to scan. Defaults to the configured documents folder.

    Returns:
        A dict mapping each processed filename to its chunk count.
    """

    if folder is None:
        folder = config.paths.documents_folder

    results: dict[str, int] = {}
    all_files = [
        p
        for p in folder.iterdir()
        if p.suffix.lower() in config.paths.supported_extensions
    ]

    for path in all_files:
        processor_cls = PROCESSORS.get(path.suffix.lower())
        if not processor_cls:
            continue

        print(f"Processing {path.name}...")
        n = processor_cls(path).process()
        results[path.name] = n

    return results


if __name__ == "__main__":
    save_all_files()
