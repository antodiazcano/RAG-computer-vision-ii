"""
Script to define how the chatbot interacts with the client.
"""

from google.genai import types

from src.config import config
from src.utils import get_gen_ai_client, get_index_vector_db


GEN_AI_CLIENT = get_gen_ai_client()
PINECONE_IDX = get_index_vector_db()


def _embed_query(question: str) -> list[float]:
    """
    Embeds the text with the embedding model.

    Args:
        text: Text to embed.

    Returns:
        Embedding of the text.
    """

    response = GEN_AI_CLIENT.models.embed_content(
        model=config.embedding_model.embedding_model,
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=config.embedding_model.embedding_dim,
        ),
    )

    return response.embeddings[0].values  # type: ignore


def _retrieve_from_vector_db(
    question: str, top_k: int = 3
) -> list[dict[str, int | str | float]]:
    """
    Retrieves the top k chunks associated with the questions of the user.

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


def _create_prompt(question: str, context: str) -> str:
    """
    Creates a prompt for the chatbot to answer.

    Args:
        question: Question of the user.
        context: Retrieved context from the vector db.

    Returns:
        Created prompt.
    """

    return (
        f"{config.chat_model.system_prompt}\n\nCONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )


def generate_answer(question: str) -> tuple[str, list[dict[str, int | str | float]]]:
    """
    Generates an answer to the question of the user using RAG.

    Args:
        question: Question of the user.

    Returns:
        Response of the model and chunks retrieved.
    """

    chunks = _retrieve_from_vector_db(question)
    context = "\n".join(
        f"[{c['source']} - Page {c['page']}: {c['text']}" for c in chunks
    )
    prompt = _create_prompt(question, context)

    response = GEN_AI_CLIENT.models.generate_content(
        model=config.chat_model.chat_model, contents=prompt
    )

    if not isinstance(response.text, str):
        raise RuntimeError("Chatbot could not answer!")

    return response.text, chunks
