"""
agent.py
--------
Controlled ReAct-style agent for the AI Study Assistant.

Stages:
1. Decide whether retrieval is required
2. Retrieve the best matching chunks
3. Validate whether chunks were retrieved
4. Produce a grounded answer or fallback response
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from utils import (
    get_openrouter_client,
    get_model_name,
    get_llm_params,
)

from rag import retrieve

from prompts import (
    SYSTEM_ROLE,
    QA_PROMPT,
    AGENT_DECIDE_RETRIEVAL,
    SUMMARY_PROMPT,
    FLASHCARD_PROMPT,
    FLASHCARD_FEW_SHOT,
    MCQ_PROMPT,
    MCQ_FEW_SHOT,
    INTERVIEW_PROMPT,
    build_context,
)


logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = (
    "The uploaded study materials do not contain enough information."
)


# ---------------------------------------------------------------------------
# Low-level LLM call
# ---------------------------------------------------------------------------

def _call_llm(
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Make one LLM request using the configured OpenRouter-compatible client.

    Returns an empty string if the request fails.
    """
    try:
        client = get_openrouter_client()
        model = get_model_name()
        params = get_llm_params()

        if temperature is not None:
            params["temperature"] = temperature

        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **params,
        )

        content = response.choices[0].message.content

        return (content or "").strip()

    except Exception as error:
        logger.exception("LLM call failed: %s", error)
        return ""


# ---------------------------------------------------------------------------
# Stage 1: Decide whether retrieval is required
# ---------------------------------------------------------------------------

def decide_retrieval(query: str) -> str:
    """
    Return either RETRIEVE or DIRECT.

    Greetings and basic system questions can be answered directly.
    Questions about uploaded documents must use retrieval.
    """
    if not query or not query.strip():
        return "DIRECT"

    prompt = AGENT_DECIDE_RETRIEVAL.format(
        query=query.strip()
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a routing agent. "
                "Reply with exactly one word: RETRIEVE or DIRECT."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    decision = _call_llm(
        messages,
        temperature=0.0,
        max_tokens=16,
    ).upper()

    if "DIRECT" in decision:
        return "DIRECT"

    # Safer default for a study assistant.
    return "RETRIEVE"


# ---------------------------------------------------------------------------
# Stage 3: Validate retrieved chunks
# ---------------------------------------------------------------------------

def validate_relevance(
    question: str,
    chunks: List[Dict[str, Any]],
) -> str:
    """
    Return SUFFICIENT when at least one chunk was retrieved.

    The final answer prompt still prevents unsupported information from
    being generated, so an additional LLM validation call is unnecessary.
    """
    if not chunks:
        return "INSUFFICIENT"

    return "SUFFICIENT"


# ---------------------------------------------------------------------------
# Stage 4: Generate grounded answer
# ---------------------------------------------------------------------------

def generate_grounded_answer(
    question: str,
    chunks: List[Dict[str, Any]],
) -> str:
    """
    Generate an answer using only the retrieved document chunks.
    """
    if not chunks:
        return FALLBACK_MESSAGE

    context = build_context(
        chunks,
        max_chars=6000,
    )

    prompt = QA_PROMPT.format(
        context=context,
        question=question,
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_ROLE,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    answer = _call_llm(
        messages,
        temperature=0.2,
        max_tokens=1024,
    )

    if not answer:
        return FALLBACK_MESSAGE

    return answer


# ---------------------------------------------------------------------------
# Main ReAct-style workflow
# ---------------------------------------------------------------------------

def run_agent(query: str) -> Dict[str, Any]:
    """
    Run the complete controlled ReAct-style workflow.

    Returns a result dictionary used by the Streamlit interface.
    """
    cleaned_query = query.strip() if query else ""

    result: Dict[str, Any] = {
        "query": cleaned_query,
        "decision": None,
        "chunks": [],
        "validation": None,
        "answer": "",
        "sources": [],
    }

    if not cleaned_query:
        result["decision"] = "DIRECT"
        result["validation"] = "INSUFFICIENT"
        result["answer"] = "Please enter a question."
        return result

    # ------------------------------------------------------------------
    # Stage 1: Decide
    # ------------------------------------------------------------------

    decision = decide_retrieval(cleaned_query)
    result["decision"] = decision

    # ------------------------------------------------------------------
    # Direct response for greetings or system questions
    # ------------------------------------------------------------------

    if decision == "DIRECT":
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the AI Study Assistant. "
                    "Respond briefly and helpfully. "
                    "Do not claim to know the uploaded document content "
                    "unless retrieval has been performed."
                ),
            },
            {
                "role": "user",
                "content": cleaned_query,
            },
        ]

        answer = _call_llm(
            messages,
            temperature=0.2,
            max_tokens=256,
        )

        result["validation"] = "NOT_REQUIRED"
        result["answer"] = answer or (
            "Hello! Upload and ingest your study materials, "
            "then ask me questions about them."
        )

        return result

    # ------------------------------------------------------------------
    # Stage 2: Retrieve
    # ------------------------------------------------------------------

    chunks = retrieve(
        query=cleaned_query,
    )

    result["chunks"] = chunks
    result["sources"] = [
        chunk.get("citation", "")
        for chunk in chunks
        if chunk.get("citation")
    ]

    # ------------------------------------------------------------------
    # Stage 3: Validate
    # ------------------------------------------------------------------

    validation = validate_relevance(
        cleaned_query,
        chunks,
    )

    result["validation"] = validation

    if validation == "INSUFFICIENT":
        result["answer"] = FALLBACK_MESSAGE
        return result

    # ------------------------------------------------------------------
    # Stage 4: Answer
    # ------------------------------------------------------------------

    answer = generate_grounded_answer(
        cleaned_query,
        chunks,
    )

    if not answer:
        result["answer"] = FALLBACK_MESSAGE
        return result

    result["answer"] = answer

    return result


# ---------------------------------------------------------------------------
# Study tool: Summary
# ---------------------------------------------------------------------------

def generate_summary(
    chunks: List[Dict[str, Any]],
) -> str:
    """
    Generate a summary from the provided chunks.
    """
    if not chunks:
        return FALLBACK_MESSAGE

    context = build_context(
        chunks,
        max_chars=6000,
    )

    prompt = SUMMARY_PROMPT.format(
        context=context
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_ROLE,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    output = _call_llm(
        messages,
        temperature=0.2,
        max_tokens=1000,
    )

    return output or FALLBACK_MESSAGE


# ---------------------------------------------------------------------------
# Study tool: Flashcards
# ---------------------------------------------------------------------------

def generate_flashcards(
    chunks: List[Dict[str, Any]],
    n: int = 5,
) -> str:
    """
    Generate flashcards using few-shot prompting.
    """
    if not chunks:
        return FALLBACK_MESSAGE

    context = build_context(
        chunks,
        max_chars=5000,
    )

    prompt = FLASHCARD_PROMPT.format(
        n=n,
        few_shot=FLASHCARD_FEW_SHOT,
        context=context,
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_ROLE,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    output = _call_llm(
        messages,
        temperature=0.3,
        max_tokens=1000,
    )

    return output or FALLBACK_MESSAGE


# ---------------------------------------------------------------------------
# Study tool: MCQs
# ---------------------------------------------------------------------------

def generate_mcqs(
    chunks: List[Dict[str, Any]],
    n: int = 5,
) -> str:
    """
    Generate multiple-choice questions using few-shot prompting.
    """
    if not chunks:
        return FALLBACK_MESSAGE

    context = build_context(
        chunks,
        max_chars=5000,
    )

    prompt = MCQ_PROMPT.format(
        n=n,
        few_shot=MCQ_FEW_SHOT,
        context=context,
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_ROLE,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    output = _call_llm(
        messages,
        temperature=0.3,
        max_tokens=1200,
    )

    return output or FALLBACK_MESSAGE


# ---------------------------------------------------------------------------
# Study tool: Interview questions
# ---------------------------------------------------------------------------

def generate_interview_questions(
    chunks: List[Dict[str, Any]],
    n: int = 5,
) -> str:
    """
    Generate interview or oral-exam questions.
    """
    if not chunks:
        return FALLBACK_MESSAGE

    context = build_context(
        chunks,
        max_chars=5000,
    )

    prompt = INTERVIEW_PROMPT.format(
        n=n,
        context=context,
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_ROLE,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    output = _call_llm(
        messages,
        temperature=0.3,
        max_tokens=900,
    )

    return output or FALLBACK_MESSAGE