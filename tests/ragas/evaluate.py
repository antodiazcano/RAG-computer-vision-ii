"""
RAGAS evaluation script for the RAG pipeline.
"""

import json
import time
from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from pydantic import SecretStr
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.evaluation import EvaluationResult
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_relevance import ResponseRelevancy
from ragas.metrics._context_recall import LLMContextRecall
from ragas.metrics._faithfulness import Faithfulness

from src.chatbot.rag import RAGTool
from src.config import config
from src.utils import get_embedding_client, get_index_vector_db


EVAL_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.json"
TOP_K = 3
DELAY_BETWEEN_QUESTIONS = 5  # seconds, to respect Groq rate limits


def _load_dataset() -> list[dict[str, str]]:
    """
    Loads the evaluation dataset.

    Returns:
        List where each element is a dict with a question and a ground truth.
    """

    return json.loads(EVAL_DATASET_PATH.read_text(encoding="utf-8"))


def _run_pipeline(
    rag_tool: RAGTool, questions: list[dict[str, str]]
) -> list[SingleTurnSample]:
    """
    Runs each question through the RAG retrieval pipeline and collects samples.

    Args:
        rag_tool: RAG engine.
        questions: List where each element is a dict with a question and a grount truth.

    Returns:
        List where each element contains the question, the responde of the LLM, the
        retrieved context by RAG and the ground truth.
    """

    samples: list[SingleTurnSample] = []

    for i, item in enumerate(questions):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"[{i + 1}/{len(questions)}] {question}")

        chunks = rag_tool.retrieve(question)
        contexts = [c["text"] for c in chunks if isinstance(c["text"], str)]

        # Use Groq to generate the answer with context
        llm = ChatGroq(
            model=config.chat_model.providers["Groq"],
            api_key=SecretStr(config.chat_model.groq_api_key or ""),
        )
        context_str = "\n".join(
            f"[{c['source']} - § {c['location']}]: {c['text']}" for c in chunks
        )
        response = llm.invoke(
            f"Based on the following context, answer the question.\n\n"
            f"CONTEXT:\n{context_str}\n\nQUESTION: {question}"
        )
        samples.append(
            SingleTurnSample(
                user_input=question,
                response=str(response.content),
                retrieved_contexts=contexts,
                reference=ground_truth,
            )
        )

        time.sleep(DELAY_BETWEEN_QUESTIONS)

    return samples


def main() -> None:
    """
    Runs the full RAGAS evaluation. It evaluates the RAG engine and the LLM.
    """

    print("Loading evaluation dataset...")
    questions = _load_dataset()
    print(f"Loaded {len(questions)} questions.\n")

    print("Initialising RAG tool...")
    embedding_client = get_embedding_client()
    pinecone_index = get_index_vector_db()
    rag_tool = RAGTool(embedding_client, pinecone_index, top_k=TOP_K)

    print("Running pipeline...\n")
    samples = _run_pipeline(rag_tool, questions)

    print("\nEvaluating with RAGAS (using Groq as judge)...\n")
    evaluator_llm = LangchainLLMWrapper(
        ChatGroq(
            model=config.chat_model.providers["Groq"],
            api_key=SecretStr(config.chat_model.groq_api_key or ""),
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model=config.embedding_model.embedding_model,
            api_key=SecretStr(config.embedding_model.api_key or ""),
        )
    )

    dataset = EvaluationDataset(samples=samples)  # type: ignore

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextRecall(),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    if not isinstance(result, EvaluationResult):
        print("Evaluation did not return results.")
        return

    scores = result.to_pandas().mean(numeric_only=True).to_dict()

    print("\n===== RAGAS Results =====")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.4f}")

    output = {
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "num_questions": len(questions),
        "top_k": TOP_K,
    }
    RESULTS_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
