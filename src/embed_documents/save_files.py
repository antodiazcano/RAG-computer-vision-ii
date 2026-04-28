"""
Script with the main functions to manage the vector db: deleting vectors and indexing
documents.
"""

import json
from collections.abc import Mapping

from pinecone.db_data.index import Index

from src.config import config
from src.embed_documents.pdf_processor import PDFProcessor
from src.embed_documents.tex_processor import TEXProcessor
from src.utils import (
    get_embedding_client,
    get_index_vector_db,
    load_registry,
    save_registry,
)


PROCESSORS = {
    ".pdf": PDFProcessor,
    ".tex": TEXProcessor,
}


def delete_vectors_by_metadata(
    pinecone_index: Index, metadata_filter: Mapping[str, str | float | int]
) -> None:
    """
    Deletes vectors from the vector db that match the given metadata filter and
    removes the corresponding entries from the indexed registry.

    Args:
        pinecone_index: Pinecone index to delete from.
        metadata_filter: Pinecone metadata filter (e.g. {'source': 'file.pdf'}).
    """

    pinecone_index.delete(filter=dict(metadata_filter))
    print(f"Deleted vectors matching {metadata_filter}.")

    reg = load_registry()
    key = metadata_filter.get("source")
    doc_type = metadata_filter.get("doc_type")

    if isinstance(key, str) and key in reg:
        del reg[key]
    elif isinstance(doc_type, str):
        ext = f".{doc_type}"
        reg = {k: v for k, v in reg.items() if not k.endswith(ext)}

    save_registry(reg)


def _build_corpus_index() -> list[dict[str, str | list[dict[str, str]]]]:
    """
    Builds a table of contents from all .tex files in the documents folder.

    Returns:
        A list of dicts, one per file, with 'source' and 'toc' keys.
    """

    corpus: list[dict[str, str | list[dict[str, str]]]] = []

    tex_files = sorted(
        p for p in config.paths.documents_folder.iterdir() if p.suffix.lower() == ".tex"
    )
    for path in tex_files:
        toc = TEXProcessor.extract_toc(path)
        corpus.append({"source": path.name, "toc": toc})

    pdf_files = sorted(
        p for p in config.paths.documents_folder.iterdir() if p.suffix.lower() == ".pdf"
    )
    for path in pdf_files:
        corpus.append({"source": path.name, "toc": []})

    return corpus


def save_all_files() -> dict[str, int]:
    """
    Indexes all supported files in a folder into the vector db and generates the corpus
    index JSON.

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

    corpus_index = _build_corpus_index()
    config.paths.corpus_index_path.write_text(
        json.dumps(corpus_index, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Corpus index saved to {config.paths.corpus_index_path}.")

    return results


if __name__ == "__main__":
    delete_vectors_by_metadata(get_index_vector_db(), {"doc_type": "tex"})
    save_all_files()
