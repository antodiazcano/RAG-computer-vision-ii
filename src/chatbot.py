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


def _create_prompt(
    question: str, context: str, chat_history: list[dict[str, str]] | None = None
) -> list[types.Content]:
    """
    Creates a prompt for the chatbot to answer.

    Args:
        question: Question of the user.
        context: Retrieved context from the vector db.
        chat_history: Previous conversation turns with "role" and "content" keys.

    Returns:
        List of Content objects representing the conversation.
    """

    contents: list[types.Content] = []
    for msg in chat_history or []:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part(text=msg["content"])])
        )

    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=f"CONTEXT:\n{context}\n\nQUESTION: {question}")],
        )
    )

    return contents


def generate_answer(
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

    chunks = _retrieve_from_vector_db(question, top_k=top_k)
    context = "\n".join(
        f"[{c['source']} - Section {c['page']}]: {c['text']}" for c in chunks
    )
    contents = _create_prompt(question, context, chat_history)

    dummy = False

    if not dummy:
        response = GEN_AI_CLIENT.models.generate_content(
            model=config.chat_model.chat_model,
            contents=contents,  # type: ignore
            config=types.GenerateContentConfig(
                system_instruction=config.chat_model.system_prompt,
            ),
        )
    else:
        return "aaa", chunks

    if not isinstance(response.text, str):
        raise RuntimeError("Chatbot could not answer!")

    return response.text, chunks
