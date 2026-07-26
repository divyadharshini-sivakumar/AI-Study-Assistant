"""
agent.py
--------
Controlled ReAct-style agent for the AI Study Assistant.

Stages:
  1. Decide → RETRIEVE or DIRECT
  2. Retrieve evidence (if needed)
  3. Validate relevance → SUFFICIENT / INSUFFICIENT
  4. Produce grounded answer or the exact fallback sentence
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from utils import get_openrouter_client, get_model_name, get_llm_params
from rag import retrieve_relevant_only, has_sufficient_evidence
from prompts import (
    SYSTEM_ROLE,
    QA_PROMPT,
    AGENT_DECIDE_RETRIEVAL,
    AGENT_VALIDATE_RELEVANCE,
    AGENT_FINAL_ANSWER,
    SUMMARY_PROMPT,
    FLASHCARD_PROMPT,
    FLASHCARD_FEW_SHOT,
    MCQ_PROMPT,
    MCQ_FEW_SHOT,
    INTERVIEW_PROMPT,
    build_context,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level LLM call
# ---------------------------------------------------------------------------

def _call_llm(
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Single OpenRouter call with free-tier friendly defaults.
    Returns the assistant content or an empty string on failure.
    """
    client = get_openrouter_client()
    model = get_model_name()
    params = get_llm_params()

    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **params,
        )
        content = response.choices[0].message.content
        return (content or "").strip()
    except Exception as e:
        logger.error("OpenRouter call failed: %s", e)
        return ""


# ---------------------------------------------------------------------------
# Stage 1 – Decide whether retrieval is required
# ---------------------------------------------------------------------------

def decide_retrieval(query: str) -> str:
    """
    Returns 'RETRIEVE' or 'DIRECT'.
    Falls back to RETRIEVE on any ambiguity (safer for a study assistant).
    """
    prompt = AGENT_DECIDE_RETRIEVAL.format(query=query)
    messages = [
        {"role": "system", "content": "You are a routing agent. Reply with one word only."},
        {"role": "user", "content": prompt},
    ]
    decision = _call_llm(messages, temperature=0.0, max_tokens=16).upper()
    if "DIRECT" in decision:
        return "DIRECT"
    return "RETRIEVE"


# ---------------------------------------------------------------------------
# Stage 3 – Validate relevance of retrieved chunks
# ---------------------------------------------------------------------------

def validate_relevance(question: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Returns 'SUFFICIENT' or 'INSUFFICIENT'.
    """
    if not chunks:
        return "INSUFFICIENT"

    # Build a compact representation for the validator
    chunk_text = "\n\n".join(
        f"[{c['citation']}]\n{c['page_content'][:400]}" for c in chunks
    )
    prompt = AGENT_VALIDATE_RELEVANCE.format(question=question, chunks=chunk_text)
    messages = [
        {"role": "system", "content": "You are a relevance validator. Reply with one word only."},
        {"role": "user", "content": prompt},
    ]
    decision = _call_llm(messages, temperature=0.0, max_tokens=16).upper()
    if "SUFFICIENT" in decision:
        return "SUFFICIENT"
    return "INSUFFICIENT"


# ---------------------------------------------------------------------------
# Stage 4 – Final grounded answer
# ---------------------------------------------------------------------------

def generate_grounded_answer(question: str, chunks: List[Dict[str, Any]]) -> str:
    """Produce the final student-facing answer."""
    context = build_context(chunks)
    prompt = QA_PROMPT.format(context=context, question=question)
    messages = [
        {"role": "system", "content": SYSTEM_ROLE},
        {"role": "user", "content": prompt},
    ]
    return _call_llm(messages)


# ---------------------------------------------------------------------------
# Main agent entry point (ReAct-style workflow)
# ---------------------------------------------------------------------------

def run_agent(query: str) -> Dict[str, Any]:
    """
    Full controlled ReAct-style loop.
    Returns a structured result dictionary for the UI.
    """
    result: Dict[str, Any] = {
        "query": query,
        "decision": None,
        "chunks": [],
        "validation": None,
        "answer": "",
        "sources": [],
    }

    # --- Stage 1: Decide ---
    decision = decide_retrieval(query)
    result["decision"] = decision

    if decision == "DIRECT":
        # Simple direct reply for greetings / meta questions
        messages = [
            {"role": "system", "content": SYSTEM_ROLE},
            {"role": "user", "content": query},
        ]
        result["answer"] = _call_llm(messages, max_tokens=256)
        return result

    # --- Stage 2: Retrieve ---
    chunks = retrieve_relevant_only(query)
    result["chunks"] = chunks
    result["sources"] = [c["citation"] for c in chunks]

    # --- Stage 3: Validate ---
    validation = validate_relevance(query, chunks)
    result["validation"] = validation

    if validation == "INSUFFICIENT":
        result["answer"] = "The uploaded study materials do not contain enough information."
        return result

    # --- Stage 4: Answer ---
    answer = generate_grounded_answer(query, chunks)
    # Final safety net
    if not answer or "do not contain enough information" in answer.lower():
        result["answer"] = "The uploaded study materials do not contain enough information."
    else:
        result["answer"] = answer

    return result


# ---------------------------------------------------------------------------
# Specialised generation helpers (used by the UI)
# ---------------------------------------------------------------------------

def generate_summary(chunks: List[Dict[str, Any]]) -> str:
    context = build_context(chunks, max_chars=4000)
    prompt = SUMMARY_PROMPT.format(context=context)
    messages = [
        {"role": "system", "content": SYSTEM_ROLE},
        {"role": "user", "content": prompt},
    ]
    return _call_llm(messages, max_tokens=800)


def generate_flashcards(chunks: List[Dict[str, Any]], n: int = 5) -> str:
    context = build_context(chunks, max_chars=3500)
    prompt = FLASHCARD_PROMPT.format(
        n=n, few_shot=FLASHCARD_FEW_SHOT, context=context
    )
    messages = [
        {"role": "system", "content": SYSTEM_ROLE},
        {"role": "user", "content": prompt},
    ]
    return _call_llm(messages, max_tokens=900)


def generate_mcqs(chunks: List[Dict[str, Any]], n: int = 5) -> str:
    context = build_context(chunks, max_chars=3500)
    prompt = MCQ_PROMPT.format(n=n, few_shot=MCQ_FEW_SHOT, context=context)
    messages = [
        {"role": "system", "content": SYSTEM_ROLE},
        {"role": "user", "content": prompt},
    ]
    return _call_llm(messages, max_tokens=1100)


def generate_interview_questions(chunks: List[Dict[str, Any]], n: int = 5) -> str:
    context = build_context(chunks, max_chars=3500)
    prompt = INTERVIEW_PROMPT.format(n=n, context=context)
    messages = [
        {"role": "system", "content": SYSTEM_ROLE},
        {"role": "user", "content": prompt},
    ]
    return _call_llm(messages, max_tokens=700)
