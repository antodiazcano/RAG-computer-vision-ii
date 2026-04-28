"""
RAG retrieval: embed query and retrieve relevant chunks from the vector db.
"""

import json

from google.genai import types
from google.genai.client import Client
from langchain_core.tools import StructuredTool
from pinecone.db_data.index import Index

from src.config import config


class RAGTool:
    """
    Encapsulates the RAG retrieval logic and exposes it as a LangChain tool.
    """

    def __init__(
        self, embedding_client: Client, pinecone_index: Index, top_k: int = 3
    ) -> None:
        """
        Constructor of the class.

        Args:
            embedding_client: Gemini client used for query embedding.
            pinecone_index: Pinecone index used for retrieval.
            top_k: Number of chunks to retrieve.
        """

        self._embedding_client = embedding_client
        self._pinecone_index = pinecone_index
        self._top_k = top_k

        self.tool = StructuredTool.from_function(
            func=self.search,
            name="search_course_material",
            description=config.chat_model.tool_rag_prompt,
        )

    def _embed_query(self, question: str) -> list[float]:
        """
        Embeds the question with the embedding model.

        Args:
            question: Question to embed.

        Returns:
            Embedding of the question.

        Raises:
            RuntimeError: If the embedding response is empty.
        """

        response = self._embedding_client.models.embed_content(
            model=config.embedding_model.embedding_model,
            contents=question,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=config.embedding_model.embedding_dim,
            ),
        )

        if (not response.embeddings) or (response.embeddings[0].values is None):
            raise RuntimeError("Embedding response is empty.")

        return response.embeddings[0].values

    def retrieve(self, question: str) -> list[dict[str, int | str | float]]:
        """
        Retrieves the top-k chunks associated with the question.

        Args:
            question: Question of the user.

        Returns:
            A list of chunk dicts with text, source, location, total_locations,
            doc_type, and score.
        """

        query_vector = self._embed_query(question)
        results = self._pinecone_index.query(
            vector=query_vector, top_k=self._top_k, include_metadata=True
        )
        return [
            {
                "text": m.metadata.get("text"),
                "source": m.metadata.get("source"),
                "location": m.metadata.get("location"),
                "total_locations": m.metadata.get("total_locations"),
                "doc_type": m.metadata.get("doc_type"),
                "score": round(float(m.score), 4),
            }
            for m in results.matches  # type: ignore
        ]

    def search(self, query: str) -> str:
        """
        Tool function: retrieves course material and returns it as JSON.

        Args:
            query: Search query.

        Returns:
            JSON string of retrieved chunks, or a message if nothing was found.
        """

        chunks = self.retrieve(query)
        if not chunks:
            return "No relevant course material found."
        return json.dumps(chunks, ensure_ascii=False)
