"""
Script to define the base class to process the documents before saving them into the
vector db.
"""

import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from google.genai import types
from google.genai.client import Client
from google.genai.errors import ClientError
from pinecone.db_data.index import Index
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.config import config
from src.utils import file_hash, load_registry, save_registry


class Processor(ABC):
    """
    Class that defines the base processor to process the documents before saving them
    into the vector db.
    """

    def __init__(
        self,
        path: Path,
        embedding_client: Client,
        pinecone_index: Index,
        batch_size: int = 50,
    ) -> None:
        """
        Constructor of the class.

        Args:
            path: Path where the file we will embed is saved.
            embedding_client: Gemini client used for embedding.
            pinecone_index: Pinecone index used for upserting vectors.
            batch_size: Size of the batch to save to the vector db.
        """

        self.path = path
        self._embedding_client = embedding_client
        self._pinecone_index = pinecone_index
        self.batch_size = batch_size

    @retry(
        retry=retry_if_exception(
            lambda e: isinstance(e, ClientError) and e.code == 429
        ),
        wait=wait_exponential(min=30, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _embed_text(self, text: str) -> list[float]:
        """
        Embeds the text with the embedding model. Retries on rate limit (429) errors
        with exponential backoff.

        Args:
            text: Text to embed.

        Returns:
            Embedding of the text.

        Raises:
            ValueError: If the embedding response is invalid (e.g. no embeddings
                returned).
        """

        response = self._embedding_client.models.embed_content(
            model=config.embedding_model.embedding_model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=config.embedding_model.embedding_dim,
            ),
        )

        if (response.embeddings is None) or (response.embeddings[0].values is None):
            raise ValueError("Invalid embedding response: no embeddings returned.")

        return response.embeddings[0].values

    @abstractmethod
    def _obtain_chunks(self) -> list[dict[str, str | int]]:
        """
        Obtains the chunks from one file.

        Returns:
            A list that for each chunk contains:
                - The text of the chunk.
                - The file where the chunk was retrieved.
                - The location of the retrieved chunk (page number or section id).
                - The total number of locations in the file.
                - The document type of the file of the chunk.
        """

    def process(self) -> int:
        """
        Saves a file into the vector db as embeddings.

        Returns:
            Number of extracted chunks.
        """

        already_included_files = load_registry()
        h = file_hash(self.path)
        if already_included_files.get(self.path.name) == h:
            print(f"Skipping {self.path} as it's already saved.")
            return 0

        chunks = self._obtain_chunks()

        vectors: list[dict[str, str | list[float] | dict[str, int | str]]] = []
        n_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        for i, c in enumerate(chunks):
            if i % self.batch_size == 0:
                print(f"Embedding batch {i // self.batch_size + 1} / {n_batches}...")
            vec_id = str(uuid.uuid4())
            vectors.append(
                {
                    "id": vec_id,
                    "values": self._embed_text(c["text"]),  # type: ignore
                    "metadata": {
                        "text": c["text"],
                        "source": c["source"],
                        "location": c["location"],
                        "total_locations": c["total_locations"],
                        "doc_type": c["doc_type"],
                    },
                }
            )

        for i in range(0, len(vectors), self.batch_size):
            batch_num = i // self.batch_size + 1
            print(f"Saving to vector db batch {batch_num} / {n_batches}...")
            self._pinecone_index.upsert(
                vectors=vectors[i : i + self.batch_size]  # type: ignore
            )

        already_included_files[self.path.name] = h
        save_registry(already_included_files)

        return len(chunks)

    def get_document_hash(self) -> str:
        """
        Gets the hash of the document.

        Returns:
            Hash of the document.
        """

        return file_hash(self.path)
