"""
Script with the main functions to manage the vector db: deleting vectors and indexing
documents.
"""

from pinecone.db_data.index import Index

from src.config import config
from src.embed_documents.pdf_processor import PDFProcessor
from src.embed_documents.tex_processor import TEXProcessor
from src.utils import get_embedding_client, get_index_vector_db


PROCESSORS = {
    ".pdf": PDFProcessor,
    ".tex": TEXProcessor,
}


def delete_vectors_by_metadata(
    pinecone_index: Index, metadata_filter: dict[str, str | float | int]
) -> None:
    """
    Deletes vectors from the vector db that match the given metadata filter.

    Args:
        pinecone_index: Pinecone index to delete from.
        metadata_filter: Pinecone metadata filter (e.g. {'source': 'file.pdf'}).
    """

    pinecone_index.delete(filter=metadata_filter)
    print(f"Deleted vectors matching {metadata_filter}.")


def save_all_files() -> dict[str, int]:
    """
    Indexes all supported files in a folder into the vector db.

    Returns:
        A dict mapping each processed filename to its chunk count.
    """

    embedding_client = get_embedding_client()
    pinecone_index = get_index_vector_db()

    results: dict[str, int] = {}
    all_files = [
        p
        for p in config.paths.documents_folder.iterdir()
        if p.suffix.lower() in config.paths.supported_extensions
    ]

    for path in all_files:
        processor_cls = PROCESSORS.get(path.suffix.lower())
        if not processor_cls:
            continue

        print(f"Processing {path.name}...")
        n = processor_cls(path, embedding_client, pinecone_index).process()
        results[path.name] = n

    return results


if __name__ == "__main__":
    delete_vectors_by_metadata(get_index_vector_db(), {"doc_type": "tex"})
    save_all_files()
