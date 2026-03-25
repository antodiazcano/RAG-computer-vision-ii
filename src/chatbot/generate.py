"""
RAG flow: embed query, retrieve relevant documents, and generate answers.
"""

from google.genai import types

from src.chatbot.clients import ChatClient
from src.config import config
from src.utils import get_embedding_client, get_index_vector_db


EMBEDDING_CLIENT = get_embedding_client()
PINECONE_IDX = get_index_vector_db()


def _embed_query(question: str) -> list[float]:
    """
    Embeds the question with the embedding model using the server's Gemini key.

    Args:
        question: Question to embed.

    Returns:
        Embedding of the question.

    Raises:
        RuntimeError: If the embedding response is empty.
    """

    response = EMBEDDING_CLIENT.models.embed_content(
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
    question: str, top_k: int = 3
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
            - The page of the retrieved chunk.
            - The total number of pages of the file of the chunk.
            - The document type of the file of the chunk.
            - The retrieval score of the chunk.
    """

    query_vector = _embed_query(question)
    results = PINECONE_IDX.query(
        vector=query_vector, top_k=top_k, include_metadata=True
    )
    return [
        {
            "text": m.metadata.get("text"),
            "source": m.metadata.get("source"),
            "page": m.metadata.get("page"),
            "total_pages": m.metadata.get("total_pages"),
            "doc_type": m.metadata.get("doc_type"),
            "score": round(float(m.score), 4),
        }
        for m in results.matches  # type: ignore
    ]


def generate_answer(
    question: str,
    chat_client: ChatClient,
    provider: str,
    top_k: int = 3,
    chat_history: list[dict[str, str]] | None = None,
) -> tuple[str, list[dict[str, int | str | float]]]:
    """
    Generates an answer to the question of the user using RAG.

    Args:
        question: Question of the user.
        chat_client: Chat client to use for generation.
        provider: Provider name to look up the model.
        top_k: Number of chunks to retrieve.
        chat_history: Previous conversation turns with "role" and "content" keys.

    Returns:
        Response of the model and chunks retrieved.
    """

    chunks = _retrieve_from_vector_db(question, top_k=top_k)
    context = "\n".join(
        f"[{c['source']} - Section {c['page']}]: {c['text']}" for c in chunks
    )

    messages = list(chat_history or [])
    messages.append(
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}
    )

    model = config.chat_model.providers[provider]
    answer = chat_client.chat(
        model=model,
        system_prompt=config.chat_model.system_prompt,
        messages=messages,
    )

    return answer, chunks
