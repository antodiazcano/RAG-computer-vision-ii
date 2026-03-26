"""
RAG flow: embed query, retrieve relevant documents, and generate answers.
"""

from google.genai import types
from google.genai.client import Client
from pinecone.db_data.index import Index

from src.chatbot.clients import ChatClient
from src.config import config


class RAGEngine:
    """
    Encapsulates the full RAG pipeline: embed, retrieve, generate.
    """

    def __init__(
        self,
        chat_client: ChatClient,
        model: str,
        embedding_client: Client,
        pinecone_index: Index,
    ) -> None:
        """
        Constructor of the class.

        Args:
            chat_client: Chat client to use for generation.
            model: Model name to use for generation.
            embedding_client: Gemini client used for query embedding.
            pinecone_index: Pinecone index used for retrieval.
        """

        self._chat_client = chat_client
        self._model = model
        self._embedding_client = embedding_client
        self._pinecone_index = pinecone_index

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

    def _retrieve_from_vector_db(
        self, question: str, top_k: int = 3
    ) -> list[dict[str, int | str | float]]:
        """
        Retrieves the top k chunks associated with the question.

        Args:
            question: Question of the user.
            top_k: Number of chunks to retrieve.

        Returns:
            A list that for each chunk retrieved contains:
                - The text of the chunk.
                - The file where the chunk was retrieved.
                - The location of the retrieved chunk (page or section).
                - The total number of locations in the file.
                - The document type of the file of the chunk.
                - The retrieval score of the chunk.
        """

        query_vector = self._embed_query(question)
        results = self._pinecone_index.query(
            vector=query_vector, top_k=top_k, include_metadata=True
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

    def generate_answer(
        self,
        question: str,
        top_k: int = 3,
        chat_history: list[dict[str, str]] | None = None,
    ) -> tuple[str, list[dict[str, int | str | float]]]:
        """
        Generates an answer to the question of the user using RAG.

        Args:
            question: Question of the user.
            top_k: Number of chunks to retrieve.
            chat_history: Previous conversation turns with "role" and "content" keys.

        Returns:
            Response of the model and chunks retrieved.
        """

        chunks = self._retrieve_from_vector_db(question, top_k=top_k)
        context = "\n".join(
            f"[{c['source']} - "
            f"{'Page' if c['doc_type'] == 'pdf' else '§'} {c['location']}]: "
            f"{c['text']}"
            for c in chunks
        )

        messages = list(chat_history or [])
        messages.append(
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}
        )

        answer = self._chat_client.chat(model=self._model, messages=messages)

        return answer, chunks
