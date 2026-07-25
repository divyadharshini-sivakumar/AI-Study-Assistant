"""
rag.py
------
Multi-document RAG engine for the AI Study Assistant.

Features:
- Local sentence-transformer embeddings
- Persistent ChromaDB storage
- Source, page, row and chunk metadata
- Similarity search with configurable threshold
- ChromaDB statistics and inspection
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from utils import (
    CHROMA_DIR,
    format_source_citation,
    get_collection_name,
    get_embedding_model_name,
    get_retrieval_settings,
    is_relevant,
    make_chunk_id,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return the LangChain-compatible Hugging Face embedding model.

    The model is downloaded during the first run and can then work locally.
    """
    model_name = get_embedding_model_name()

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ---------------------------------------------------------------------------
# ChromaDB vector store
# ---------------------------------------------------------------------------

def get_vectorstore(create_if_missing: bool = True) -> Chroma:
    """
    Connect to the persistent ChromaDB vector store.

    The create_if_missing parameter is kept for compatibility with the
    remaining project code. Chroma automatically creates the collection
    when it does not already exist.
    """
    return Chroma(
        collection_name=get_collection_name(),
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


# ---------------------------------------------------------------------------
# Collection statistics
# ---------------------------------------------------------------------------

def get_collection_stats() -> Dict[str, Any]:
    """
    Return information about the current ChromaDB collection.

    Returns:
        collection_name
        document_count
        embedding_dimension
        persist_directory
    """
    try:
        vectorstore = get_vectorstore()
        collection = vectorstore._collection

        document_count = collection.count()
        embedding_dimension = None

        # Only inspect embeddings when the collection contains documents.
        if document_count > 0:
            peek_result = collection.peek(limit=1)

            embeddings = (
                peek_result.get("embeddings")
                if peek_result is not None
                else None
            )

            # Chroma may return embeddings as a NumPy array.
            # Therefore, check its length instead of using it directly
            # inside an if condition.
            if embeddings is not None and len(embeddings) > 0:
                first_embedding = embeddings[0]

                if first_embedding is not None:
                    embedding_dimension = len(first_embedding)

        return {
            "collection_name": get_collection_name(),
            "document_count": document_count,
            "embedding_dimension": embedding_dimension,
            "persist_directory": str(CHROMA_DIR),
        }

    except Exception as error:
        logger.exception("Could not read ChromaDB collection statistics.")

        return {
            "collection_name": get_collection_name(),
            "document_count": 0,
            "embedding_dimension": None,
            "persist_directory": str(CHROMA_DIR),
            "error": str(error),
        }


# ---------------------------------------------------------------------------
# Collection inspection
# ---------------------------------------------------------------------------

def inspect_collection(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Return stored chunks for the ChromaDB inspection section.

    Each returned item contains:
    - ID
    - metadata
    - text preview
    - embedding dimension
    """
    try:
        vectorstore = get_vectorstore()
        collection = vectorstore._collection

        document_count = collection.count()

        if document_count == 0:
            return []

        safe_limit = max(1, min(limit, document_count))

        results = collection.get(
            limit=safe_limit,
            include=["documents", "metadatas", "embeddings"],
        )

        ids = results.get("ids") or []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        embeddings = results.get("embeddings")

        items: List[Dict[str, Any]] = []

        for index, document_id in enumerate(ids):
            text = (
                documents[index]
                if index < len(documents) and documents[index] is not None
                else ""
            )

            metadata = (
                metadatas[index]
                if index < len(metadatas) and metadatas[index] is not None
                else {}
            )

            embedding = None

            if embeddings is not None and index < len(embeddings):
                embedding = embeddings[index]

            text_preview = text

            if len(text) > 300:
                text_preview = text[:300] + "…"

            items.append(
                {
                    "id": document_id,
                    "metadata": metadata,
                    "text_preview": text_preview,
                    "embedding_dim": (
                        len(embedding)
                        if embedding is not None
                        else None
                    ),
                }
            )

        return items

    except Exception:
        logger.exception("ChromaDB inspection failed.")
        return []


# ---------------------------------------------------------------------------
# Add documents
# ---------------------------------------------------------------------------

def add_documents(documents: List[Document]) -> int:
    """
    Add LangChain documents to ChromaDB.

    Returns:
        Number of document chunks added.
    """
    if not documents:
        return 0

    vectorstore = get_vectorstore()

    ids: List[str] = []

    for index, document in enumerate(documents):
        source = document.metadata.get("source", "unknown")

        chunk_id = document.metadata.get("chunk_id")

        if not chunk_id:
            chunk_id = make_chunk_id(source, index)
            document.metadata["chunk_id"] = chunk_id

        ids.append(chunk_id)

    vectorstore.add_documents(
        documents=documents,
        ids=ids,
    )

    logger.info("Added %s chunks to ChromaDB.", len(documents))

    return len(documents)


# ---------------------------------------------------------------------------
# Clear vector database
# ---------------------------------------------------------------------------

def clear_vectorstore() -> None:
    """
    Delete the current ChromaDB collection and recreate it as empty.
    """
    try:
        vectorstore = get_vectorstore()
        vectorstore.delete_collection()

    except Exception as error:
        logger.warning(
            "The collection could not be deleted or did not exist: %s",
            error,
        )

    # Accessing the vector store again recreates the collection.
    get_vectorstore()

    logger.info("ChromaDB collection cleared and recreated.")


# ---------------------------------------------------------------------------
# Similarity retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Perform similarity search in ChromaDB.

    Chroma returns a distance score:
    - Lower distance means better similarity.
    - Higher distance means weaker similarity.
    """
    if not query or not query.strip():
        return []

    default_top_k, default_threshold = get_retrieval_settings()

    if top_k is None:
        top_k = default_top_k

    if score_threshold is None:
        score_threshold = default_threshold

    try:
        vectorstore = get_vectorstore()

        collection_count = vectorstore._collection.count()

        if collection_count == 0:
            logger.info("Retrieval skipped because ChromaDB is empty.")
            return []

        safe_top_k = max(1, min(top_k, collection_count))

        results = vectorstore.similarity_search_with_score(
            query=query.strip(),
            k=safe_top_k,
        )

    except Exception:
        logger.exception("ChromaDB retrieval failed.")
        return []

    formatted_results: List[Dict[str, Any]] = []

    for document, score in results:
        numeric_score = float(score)

        formatted_results.append(
            {
                "page_content": document.page_content,
                "metadata": document.metadata,
                "score": numeric_score,
                "is_relevant": is_relevant(
                    numeric_score,
                    score_threshold,
                ),
                "citation": format_source_citation(
                    document.metadata
                ),
            }
        )

    return formatted_results


def retrieve_relevant_only(
    query: str,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve documents and keep only chunks that pass the relevance threshold.
    """
    results = retrieve(
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
    )

    return [
        result
        for result in results
        if result.get("is_relevant", False)
    ]


def has_sufficient_evidence(
    query: str,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Check whether at least one relevant document chunk was retrieved.
    """
    relevant_chunks = retrieve_relevant_only(query)

    return len(relevant_chunks) > 0, relevant_chunks