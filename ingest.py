"""
ingest.py
---------
Document loading, chunking and ingestion pipeline.
Supports PDF, TXT and CSV study materials.
Preserves rich metadata for every chunk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from utils import (
    DATA_DIR,
    get_chunk_settings,
    sanitize_filename,
    make_chunk_id,
)
from rag import add_documents, clear_vectorstore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_pdf(file_path: Path) -> List[Document]:
    """Load a PDF and attach source + page metadata."""
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()
    source = file_path.name
    for doc in docs:
        doc.metadata["source"] = source
        doc.metadata["file_type"] = "pdf"
        # page is already present from PyPDFLoader
    return docs


def load_txt(file_path: Path) -> List[Document]:
    """Load a plain-text file."""
    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()
    source = file_path.name
    for doc in docs:
        doc.metadata["source"] = source
        doc.metadata["file_type"] = "txt"
        doc.metadata["page"] = None
    return docs


def load_csv(file_path: Path) -> List[Document]:
    """
    Load a CSV. Each row becomes a separate Document so that
    row-level provenance is preserved.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error("Failed to read CSV %s: %s", file_path, e)
        return []

    source = file_path.name
    docs: List[Document] = []
    for idx, row in df.iterrows():
        # Turn the row into a readable text block
        text_parts = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
        page_content = " | ".join(text_parts)
        if not page_content.strip():
            continue
        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "source": source,
                    "file_type": "csv",
                    "page": None,
                    "row": int(idx) + 1,  # 1-based for humans
                },
            )
        )
    return docs


def load_document(file_path: Path) -> List[Document]:
    """Dispatch to the correct loader based on suffix."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(file_path)
    if suffix == ".txt":
        return load_txt(file_path)
    if suffix == ".csv":
        return load_csv(file_path)
    logger.warning("Unsupported file type: %s", file_path)
    return []


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks while preserving and enriching metadata.
    """
    chunk_size, chunk_overlap = get_chunk_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Enrich every chunk with a stable chunk_id
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        chunk.metadata["chunk_id"] = make_chunk_id(source, i)
        # Ensure keys always exist
        chunk.metadata.setdefault("page", None)
        chunk.metadata.setdefault("row", None)
        chunk.metadata.setdefault("file_type", "unknown")

    return chunks


# ---------------------------------------------------------------------------
# Public ingestion API
# ---------------------------------------------------------------------------

def ingest_files(file_paths: List[Path], clear_first: bool = False) -> dict:
    """
    Full ingestion pipeline.
    Returns a summary dictionary.
    """
    if clear_first:
        clear_vectorstore()

    all_chunks: List[Document] = []
    file_summaries = []

    for path in file_paths:
        path = Path(path)
        if not path.exists():
            logger.warning("File not found: %s", path)
            continue

        raw_docs = load_document(path)
        if not raw_docs:
            file_summaries.append(
                {"file": path.name, "status": "failed", "chunks": 0}
            )
            continue

        chunks = split_documents(raw_docs)
        all_chunks.extend(chunks)
        file_summaries.append(
            {
                "file": path.name,
                "status": "success",
                "raw_docs": len(raw_docs),
                "chunks": len(chunks),
            }
        )

    added = 0
    if all_chunks:
        added = add_documents(all_chunks)

    return {
        "files_processed": len(file_summaries),
        "total_chunks_added": added,
        "details": file_summaries,
    }


def ingest_uploaded_files(uploaded_files, clear_first: bool = False) -> dict:
    """
    Convenience wrapper for Streamlit UploadedFile objects.
    Saves them temporarily under data/ then runs the normal pipeline.
    """
    saved_paths: List[Path] = []
    for uf in uploaded_files:
        safe_name = sanitize_filename(uf.name)
        dest = DATA_DIR / safe_name
        with open(dest, "wb") as f:
            f.write(uf.getbuffer())
        saved_paths.append(dest)

    return ingest_files(saved_paths, clear_first=clear_first)


def rebuild_knowledge_base(file_paths: Optional[List[Path]] = None) -> dict:
    """
    Clear the vector store and re-ingest either the supplied files
    or every supported file currently present in data/.
    """
    if file_paths is None:
        file_paths = (
            list(DATA_DIR.glob("*.pdf"))
            + list(DATA_DIR.glob("*.txt"))
            + list(DATA_DIR.glob("*.csv"))
        )
    return ingest_files(file_paths, clear_first=True)
