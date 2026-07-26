"""
agent.py

ReAct-style AI agent for the AI Study Assistant.

Workflow:
1. Decide whether document retrieval is required.
2. Retrieve relevant chunks from ChromaDB.
3. Validate the retrieved context.
4. Generate an answer using only the retrieved study materials.
"""

from typing import Any, Dict, List

from rag import retrieve
from utils import get_llm


FALLBACK_MESSAGE = (
    "The uploaded study materials do not contain enough information."
)


def _extract_chunk_text(chunk: Dict[str, Any]) -> str:
    """
    Extract document text from a retrieved chunk.

    This supports different possible keys returned by rag.py.
    """
    return str(
        chunk.get("document")
        or chunk.get("content")
        or chunk.get("text")
        or chunk.get("page_content")
        or ""
    ).strip()


def _format_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Combine retrieved ChromaDB chunks into one context block.
    """
    context_parts: List[str] = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        text = _extract_chunk_text(chunk)

        if not text:
            continue

        metadata = chunk.get("metadata") or {}

        source = (
            metadata.get("source")
            or metadata.get("filename")
            or metadata.get("file_name")
            or chunk.get("source")
            or "Uploaded study material"
        )

        page = (
            metadata.get("page")
            if metadata.get("page") is not None
            else chunk.get("page", "N/A")
        )

        context_parts.append(
            f"""
SOURCE {index}
File: {source}
Page: {page}

{text}
""".strip()
        )

    return "\n\n---\n\n".join(context_parts)


def _invoke_llm(prompt: str) -> str:
    """
    Invoke the configured OpenRouter LLM and return plain text.
    """
    llm = get_llm()
    response = llm.invoke(prompt)

    if hasattr(response, "content"):
        return str(response.content).strip()

    return str(response).strip()


def decide_action(question: str) -> str:
    """
    Decide whether the question requires retrieval.

    Since this application answers questions from uploaded documents,
    most meaningful study questions should use retrieval.
    """
    if not question or not question.strip():
        return "NO_QUESTION"

    return "RETRIEVE"


def generate_answer(question: str, context: str) -> str:
    """
    Generate the final answer strictly from retrieved context.
    """
    prompt = f"""
You are an AI Study Assistant.

Answer the user's question using only the uploaded study-material context
provided below.

Important rules:

1. Use only information found in the provided context.
2. Do not use outside knowledge.
3. Do not invent facts.
4. Give a clear and complete answer.
5. When the context directly or indirectly contains the answer, answer it.
6. Do not return the fallback message merely because the wording in the
   question is different from the wording in the context.
7. The context may contain the answer across multiple retrieved chunks.
8. Combine relevant information from the chunks when necessary.
9. Only return the exact sentence below when the context genuinely contains
   no information that can answer the question:

"{FALLBACK_MESSAGE}"

Uploaded study-material context:

{context}

User question:

{question}

Answer:
"""

    answer = _invoke_llm(prompt)

    if not answer:
        return FALLBACK_MESSAGE

    return answer


def run_agent(question: str) -> Dict[str, Any]:
    """
    Run the complete ReAct-style document question-answering workflow.

    Returns:
        {
            "answer": str,
            "sources": list,
            "trace": {
                "decision": str,
                "validation": str,
                "chunks_retrieved": int
            }
        }
    """
    cleaned_question = question.strip() if question else ""

    trace: Dict[str, Any] = {
        "decision": "NO_QUESTION",
        "validation": "INSUFFICIENT",
        "chunks_retrieved": 0,
    }

    if not cleaned_question:
        return {
            "answer": "Please enter a question.",
            "sources": [],
            "trace": trace,
        }

    # Step 1: Decide
    decision = decide_action(cleaned_question)
    trace["decision"] = decision

    if decision != "RETRIEVE":
        return {
            "answer": FALLBACK_MESSAGE,
            "sources": [],
            "trace": trace,
        }

    # Step 2: Retrieve from ChromaDB
    try:
        retrieved_chunks = retrieve(cleaned_question)
    except Exception as error:
        return {
            "answer": (
                "An error occurred while retrieving the uploaded study "
                f"materials: {error}"
            ),
            "sources": [],
            "trace": trace,
        }

    if retrieved_chunks is None:
        retrieved_chunks = []

    trace["chunks_retrieved"] = len(retrieved_chunks)

    # Step 3: Validate deterministically
    #
    # Do not ask the LLM to reject retrieved chunks again.
    # If ChromaDB returned usable text, treat the context as sufficient.
    usable_chunks = [
        chunk
        for chunk in retrieved_chunks
        if isinstance(chunk, dict) and _extract_chunk_text(chunk)
    ]

    if not usable_chunks:
        trace["validation"] = "INSUFFICIENT"

        return {
            "answer": FALLBACK_MESSAGE,
            "sources": retrieved_chunks,
            "trace": trace,
        }

    trace["validation"] = "SUFFICIENT"

    # Step 4: Build context
    context = _format_context(usable_chunks)

    if not context.strip():
        trace["validation"] = "INSUFFICIENT"

        return {
            "answer": FALLBACK_MESSAGE,
            "sources": retrieved_chunks,
            "trace": trace,
        }

    # Step 5: Generate final answer
    try:
        answer = generate_answer(cleaned_question, context)
    except Exception as error:
        answer = (
            "An error occurred while generating the answer: "
            f"{error}"
        )

    return {
        "answer": answer,
        "sources": retrieved_chunks,
        "trace": trace,
    }


def answer_question(question: str) -> Dict[str, Any]:
    """
    Compatibility function in case app.py imports answer_question().
    """
    return run_agent(question)


def ask_agent(question: str) -> Dict[str, Any]:
    """
    Compatibility function in case app.py imports ask_agent().
    """
    return run_agent(question)
