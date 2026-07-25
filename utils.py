"""
utils.py
--------
Shared helpers for the AI Study Assistant.
All configuration is loaded from environment variables.
Never hard-code API keys.
"""

from __future__ import annotations

import os
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()

def get_env(key: str, default: Optional[str] = None, required: bool = False) -> str:
    """Fetch an environment variable with optional default and required check."""
    value = os.getenv(key, default)
    if required and (value is None or value.strip() == ""):
        raise ValueError(
            f"Required environment variable '{key}' is missing. "
            "Copy .env.example → .env and fill in the values."
        )
    return value or ""


# ---------------------------------------------------------------------------
# OpenRouter client (OpenAI-compatible)
# ---------------------------------------------------------------------------

def get_openrouter_client() -> OpenAI:
    """
    Create an OpenAI-compatible client pointed at OpenRouter.
    Key is read from environment only.
    """
    api_key = get_env("OPENROUTER_API_KEY", required=True)
    base_url = get_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={
            "HTTP-Referer": "https://github.com/your-username/AI-Study-Assistant",
            "X-Title": "AI Study Assistant",
        },
    )


def get_model_name() -> str:
    """Return the currently configured free model."""
    return get_env(
        "OPENROUTER_MODEL",
        "meta-llama/llama-3.1-8b-instruct:free",
    )


def get_llm_params() -> Dict[str, Any]:
    """Temperature and max_tokens tuned for free-tier limits."""
    return {
        "temperature": float(get_env("OPENROUTER_TEMPERATURE", "0.2")),
        "max_tokens": int(get_env("OPENROUTER_MAX_TOKENS", "1024")),
    }


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = Path(get_env("CHROMA_PERSIST_DIRECTORY", "./chroma_db"))
REPORTS_DIR = PROJECT_ROOT / "reports"
ASSETS_DIR = PROJECT_ROOT / "assets"

# Ensure directories exist
for d in (DATA_DIR, CHROMA_DIR, REPORTS_DIR, ASSETS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def get_collection_name() -> str:
    return get_env("CHROMA_COLLECTION_NAME", "study_materials")


def get_chunk_settings() -> Tuple[int, int]:
    size = int(get_env("CHUNK_SIZE", "800"))
    overlap = int(get_env("CHUNK_OVERLAP", "120"))
    return size, overlap


def get_retrieval_settings() -> Tuple[int, float]:
    top_k = int(get_env("TOP_K", "4"))
    threshold = float(get_env("SIMILARITY_THRESHOLD", "0.35"))
    return top_k, threshold


# ---------------------------------------------------------------------------
# Embedding model (local, free)
# ---------------------------------------------------------------------------

_embedding_model = None


def get_embedding_model_name() -> str:
    return get_env(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )


def get_embedding_model():
    """
    Lazy-load and cache the sentence-transformers model.
    Runs entirely offline after first download.
    """
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        model_name = get_embedding_model_name()
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """Generate a single query embedding."""
    return embed_texts([query])[0]


# ---------------------------------------------------------------------------
# Text & source helpers
# ---------------------------------------------------------------------------

def sanitize_filename(name: str) -> str:
    """Make a filename safe for storage."""
    name = re.sub(r"[^\w\s.-]", "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:120] or "unnamed"


def make_chunk_id(source: str, chunk_index: int) -> str:
    """Deterministic chunk identifier."""
    raw = f"{source}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def format_source_citation(metadata: Dict[str, Any]) -> str:
    """Human-readable source string for UI display."""
    source = metadata.get("source", "unknown")
    file_type = metadata.get("file_type", "")
    page = metadata.get("page")
    row = metadata.get("row")
    chunk_id = metadata.get("chunk_id", "")

    parts = [f"**{source}**"]
    if file_type:
        parts.append(f"({file_type})")
    if page is not None:
        parts.append(f"page {page}")
    if row is not None:
        parts.append(f"row {row}")
    if chunk_id:
        parts.append(f"chunk `{chunk_id}`")
    return " · ".join(parts)


def is_relevant(score: float, threshold: float | None = None) -> bool:
    """
    Chroma returns distance (lower = more similar).
    We convert to a simple relevance check.
    """
    if threshold is None:
        _, threshold = get_retrieval_settings()
    # For cosine distance: 0 = identical, 2 = opposite.
    # We treat scores below threshold as relevant.
    return score <= threshold


# ---------------------------------------------------------------------------
# Streamlit session helpers (lightweight)
# ---------------------------------------------------------------------------

def init_session_state(defaults: Dict[str, Any]) -> None:
    """Initialise Streamlit session_state keys if missing."""
    import streamlit as st
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_session_keys(keys: List[str]) -> None:
    """Remove specific keys from session_state."""
    import streamlit as st
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]
