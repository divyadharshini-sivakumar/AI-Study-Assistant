"""
app.py
------
AI Study Assistant – Streamlit frontend.
Modern, student-friendly UI with full RAG + ReAct agent capabilities.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import chromadb
import streamlit as st

from utils import (
    init_session_state,
    clear_session_keys,
    CHROMA_DIR,
)

from rag import (
    get_collection_stats,
    inspect_collection,
    clear_vectorstore,
)

from ingest import (
    ingest_uploaded_files,
    rebuild_knowledge_base,
)

from agent import (
    run_agent,
    generate_summary,
    generate_flashcards,
    generate_mcqs,
    generate_interview_questions,
)

from report_generator import generate_report


# ---------------------------------------------------------------------------
# Page configuration and logging
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FALLBACK_MESSAGE = (
    "The uploaded study materials do not contain enough information."
)


# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------

DEFAULTS = {
    "messages": [],
    "last_agent_result": None,
    "ingestion_log": None,
    "study_tool_output": None,
    "study_tool_type": None,
    "show_chromadb_backend": False,
}

init_session_state(DEFAULTS)


# ---------------------------------------------------------------------------
# Helper: Read ChromaDB backend
# ---------------------------------------------------------------------------

def get_chromadb_backend_data(limit: int = 5) -> Dict[str, Any]:
    """
    Connect to the existing ChromaDB database and return human-readable
    collection information.

    This is only used for displaying backend data in the Streamlit UI.
    """
    result: Dict[str, Any] = {
        "database_path": str(CHROMA_DIR),
        "collections": [],
        "error": None,
    }

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collections = client.list_collections()

        for collection_info in collections:
            collection_name = collection_info.name

            collection = client.get_collection(
                name=collection_name
            )

            total_chunks = collection.count()

            # Retrieve a small number of records for display.
            data = collection.get(
                limit=limit,
                include=[
                    "documents",
                    "metadatas",
                    "embeddings",
                ],
            )

            ids = data.get("ids") or []
            documents = data.get("documents") or []
            metadatas = data.get("metadatas") or []
            embeddings = data.get("embeddings")

            records: List[Dict[str, Any]] = []

            for index, chunk_id in enumerate(ids):
                document = (
                    documents[index]
                    if index < len(documents)
                    else ""
                )

                metadata = (
                    metadatas[index]
                    if index < len(metadatas)
                    else {}
                )

                embedding_dimension = None

                if embeddings is not None and index < len(embeddings):
                    current_embedding = embeddings[index]

                    if current_embedding is not None:
                        embedding_dimension = len(current_embedding)

                records.append(
                    {
                        "id": chunk_id,
                        "document": document,
                        "metadata": metadata,
                        "embedding_dimension": embedding_dimension,
                    }
                )

            result["collections"].append(
                {
                    "name": collection_name,
                    "total_chunks": total_chunks,
                    "records": records,
                }
            )

    except Exception as error:
        logger.exception(
            "Unable to inspect ChromaDB backend: %s",
            error,
        )
        result["error"] = str(error)

    return result


# ---------------------------------------------------------------------------
# Helper: Display retrieved sources as clickable expandable cards
# ---------------------------------------------------------------------------

def display_retrieved_sources(
    chunks: List[Dict[str, Any]],
    key_prefix: str,
) -> None:
    """
    Display retrieved sources as simple clickable expanders.

    Clicking a source shows only the retrieved document content.
    """
    if not chunks:
        return

    st.markdown("#### 📚 Retrieved sources")

    for index, chunk in enumerate(chunks, start=1):
        citation = chunk.get("citation", "Unknown source")
        page_content = chunk.get("page_content", "")

        with st.expander(
            f"Source {index}: {citation}",
            expanded=False,
        ):
            if page_content:
                st.markdown(page_content)
            else:
                st.info("No text is available for this source.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📚 AI Study Assistant")
    st.caption(
        "Multi-document RAG · ReAct Agent · Free-tier ready"
    )

    st.divider()
    st.subheader("📁 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload study materials",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True,
        help="PDF, TXT and CSV files are supported.",
    )

    col1, col2 = st.columns(2)

    with col1:
        ingest_btn = st.button(
            "➕ Ingest",
            use_container_width=True,
            type="primary",
        )

    with col2:
        clear_btn = st.button(
            "🗑️ Clear DB",
            use_container_width=True,
        )

    rebuild_btn = st.button(
        "🔄 Rebuild from data/",
        use_container_width=True,
    )

    # -----------------------------------------------------------------------
    # Ingest uploaded files
    # -----------------------------------------------------------------------

    if ingest_btn:
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            with st.spinner("Ingesting documents..."):
                try:
                    result = ingest_uploaded_files(
                        uploaded_files,
                        clear_first=False,
                    )

                    st.session_state.ingestion_log = result

                    st.success(
                        f"Added {result['total_chunks_added']} chunks "
                        f"from {result['files_processed']} file(s)."
                    )

                    st.rerun()

                except Exception as error:
                    logger.exception(
                        "Document ingestion failed: %s",
                        error,
                    )
                    st.error(
                        f"Document ingestion failed: {error}"
                    )

    # -----------------------------------------------------------------------
    # Clear database
    # -----------------------------------------------------------------------

    if clear_btn:
        try:
            clear_vectorstore()

            clear_session_keys(
                [
                    "messages",
                    "last_agent_result",
                    "study_tool_output",
                ]
            )

            st.session_state.ingestion_log = {
                "status": "cleared"
            }

            st.session_state.show_chromadb_backend = False

            st.success("Vector database cleared.")
            st.rerun()

        except Exception as error:
            logger.exception(
                "Unable to clear vector database: %s",
                error,
            )
            st.error(
                f"Unable to clear vector database: {error}"
            )

    # -----------------------------------------------------------------------
    # Rebuild database
    # -----------------------------------------------------------------------

    if rebuild_btn:
        with st.spinner(
            "Rebuilding knowledge base from data/..."
        ):
            try:
                result = rebuild_knowledge_base()

                st.session_state.ingestion_log = result

                st.success(
                    f"Rebuilt: "
                    f"{result['total_chunks_added']} chunks."
                )

                st.rerun()

            except Exception as error:
                logger.exception(
                    "Knowledge-base rebuild failed: %s",
                    error,
                )
                st.error(
                    f"Knowledge-base rebuild failed: {error}"
                )

    # -----------------------------------------------------------------------
    # Live collection statistics
    # -----------------------------------------------------------------------

    stats = get_collection_stats()

    st.divider()
    st.subheader("📊 Collection Stats")

    st.metric(
        "Documents / Chunks",
        stats.get("document_count", 0),
    )

    st.caption(
        f"Collection: "
        f"`{stats.get('collection_name', '—')}`"
    )

    if stats.get("embedding_dimension"):
        st.caption(
            f"Embedding dimension: "
            f"{stats['embedding_dimension']}"
        )

    # -----------------------------------------------------------------------
    # ChromaDB backend button
    # -----------------------------------------------------------------------

    st.divider()
    st.subheader("🗄️ Backend Database")

    if st.button(
        "View ChromaDB Backend",
        use_container_width=True,
    ):
        st.session_state.show_chromadb_backend = True

    if st.session_state.show_chromadb_backend:
        if st.button(
            "Hide Backend Data",
            use_container_width=True,
        ):
            st.session_state.show_chromadb_backend = False
            st.rerun()

    # -----------------------------------------------------------------------
    # Study tools
    # -----------------------------------------------------------------------

    st.divider()
    st.subheader("🛠️ Study Tools")

    tool = st.selectbox(
        "Choose a tool",
        [
            "None",
            "Summary",
            "Flashcards",
            "MCQs",
            "Interview Questions",
        ],
        index=0,
    )

    n_items = st.slider(
        "Number of items",
        min_value=3,
        max_value=10,
        value=5,
    )

    if st.button(
        "Generate",
        use_container_width=True,
    ) and tool != "None":

        if stats.get("document_count", 0) == 0:
            st.warning(
                "Knowledge base is empty. "
                "Upload materials first."
            )

        else:
            from rag import retrieve_relevant_only

            chunks = retrieve_relevant_only(
                "key concepts definitions important points",
                top_k=8,
            )

            if not chunks:
                st.warning("No relevant chunks found.")

            else:
                with st.spinner(
                    f"Generating {tool.lower()}..."
                ):
                    if tool == "Summary":
                        output = generate_summary(chunks)

                    elif tool == "Flashcards":
                        output = generate_flashcards(
                            chunks,
                            n=n_items,
                        )

                    elif tool == "MCQs":
                        output = generate_mcqs(
                            chunks,
                            n=n_items,
                        )

                    else:
                        output = (
                            generate_interview_questions(
                                chunks,
                                n=n_items,
                            )
                        )

                    st.session_state.study_tool_output = output
                    st.session_state.study_tool_type = tool

    # -----------------------------------------------------------------------
    # PDF report
    # -----------------------------------------------------------------------

    st.divider()

    if st.button(
        "📄 Generate PDF Report",
        use_container_width=True,
    ):
        with st.spinner("Building report..."):
            try:
                report_path = generate_report()

                st.success(
                    f"Report saved to "
                    f"`{report_path.name}`"
                )

                with open(report_path, "rb") as report_file:
                    st.download_button(
                        label="Download Report",
                        data=report_file,
                        file_name=report_path.name,
                        mime="application/pdf",
                        use_container_width=True,
                    )

            except Exception as error:
                logger.exception(
                    "Report generation failed: %s",
                    error,
                )
                st.error(
                    f"Report generation failed: {error}"
                )


# ---------------------------------------------------------------------------
# Main application area
# ---------------------------------------------------------------------------

st.title("Ask your study materials")

st.caption(
    "Answers are grounded exclusively in the documents "
    "you uploaded."
)


# ---------------------------------------------------------------------------
# Ingestion feedback
# ---------------------------------------------------------------------------

if st.session_state.ingestion_log:
    with st.expander(
        "Last ingestion details",
        expanded=False,
    ):
        st.json(
            st.session_state.ingestion_log
        )


# ---------------------------------------------------------------------------
# ChromaDB backend viewer
# ---------------------------------------------------------------------------

if st.session_state.show_chromadb_backend:
    st.divider()
    st.header("🗄️ ChromaDB Backend Database")

    st.caption(
        "This section displays the data stored in the "
        "ChromaDB vector database."
    )

    display_limit = st.number_input(
        "Number of stored chunks to display",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        key="backend_display_limit",
    )

    backend_data = get_chromadb_backend_data(
        limit=int(display_limit)
    )

    if backend_data.get("error"):
        st.error(
            "Unable to open the ChromaDB backend."
        )
        st.code(
            backend_data["error"]
        )

    else:
        st.info(
            f"Database location: "
            f"`{backend_data['database_path']}`"
        )

        collections = backend_data.get(
            "collections",
            [],
        )

        if not collections:
            st.warning(
                "No ChromaDB collections were found. "
                "Upload and ingest documents first."
            )

        else:
            for collection in collections:
                collection_name = collection["name"]
                total_chunks = collection["total_chunks"]
                records = collection["records"]

                st.subheader(
                    f"Collection: `{collection_name}`"
                )

                metric_col1, metric_col2 = st.columns(2)

                with metric_col1:
                    st.metric(
                        "Total Stored Chunks",
                        total_chunks,
                    )

                with metric_col2:
                    embedding_dimension = "—"

                    if records:
                        first_dimension = records[0].get(
                            "embedding_dimension"
                        )

                        if first_dimension:
                            embedding_dimension = (
                                first_dimension
                            )

                    st.metric(
                        "Embedding Dimension",
                        embedding_dimension,
                    )

                st.markdown(
                    "### Stored Document Records"
                )

                if not records:
                    st.warning(
                        "The collection exists, but it "
                        "does not contain any records."
                    )

                for record_number, record in enumerate(
                    records,
                    start=1,
                ):
                    record_id = record.get(
                        "id",
                        "Unknown ID",
                    )

                    with st.expander(
                        f"Record {record_number} — "
                        f"ID: {record_id}"
                    ):
                        st.markdown("#### Chunk ID")
                        st.code(record_id)

                        st.markdown("#### Metadata")

                        metadata = record.get(
                            "metadata",
                            {},
                        )

                        if metadata:
                            st.json(metadata)
                        else:
                            st.info(
                                "No metadata is available "
                                "for this chunk."
                            )

                        st.markdown(
                            "#### Stored Document Chunk"
                        )

                        document_text = record.get(
                            "document",
                            "",
                        )

                        if document_text:
                            st.text_area(
                                label=(
                                    f"Chunk {record_number} text"
                                ),
                                value=document_text,
                                height=180,
                                disabled=True,
                                label_visibility="collapsed",
                                key=(
                                    f"backend_chunk_"
                                    f"{collection_name}_"
                                    f"{record_number}"
                                ),
                            )
                        else:
                            st.info(
                                "No text is stored for "
                                "this record."
                            )

                        if record.get(
                            "embedding_dimension"
                        ):
                            st.caption(
                                "Embedding dimension: "
                                f"{record['embedding_dimension']}"
                            )

    st.divider()


# ---------------------------------------------------------------------------
# Study-tool output
# ---------------------------------------------------------------------------

if st.session_state.study_tool_output:
    st.subheader(
        f"📝 {st.session_state.study_tool_type}"
    )

    st.markdown(
        st.session_state.study_tool_output
    )

    st.divider()


# ---------------------------------------------------------------------------
# Display previous chat messages
# ---------------------------------------------------------------------------

for message_index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("chunks"):
            display_retrieved_sources(
                message["chunks"],
                key_prefix=f"history_{message_index}",
            )
        elif message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input(
    "Ask a question about your materials..."
):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = run_agent(prompt)

                st.session_state.last_agent_result = result

                answer = result.get(
                    "answer",
                    FALLBACK_MESSAGE,
                )

                st.markdown(answer)

                sources = result.get(
                    "sources",
                    [],
                )

                if sources:
                    display_retrieved_sources(
                        result.get("chunks", []),
                        key_prefix=f"current_{len(st.session_state.messages)}",
                    )

                with st.expander(
                    "Agent trace (ReAct stages)"
                ):
                    st.write(
                        f"**Decision:** "
                        f"{result.get('decision')}"
                    )

                    st.write(
                        f"**Validation:** "
                        f"{result.get('validation')}"
                    )

                    st.write(
                        f"**Chunks retrieved:** "
                        f"{len(result.get('chunks', []))}"
                    )

            except Exception as error:
                logger.exception(
                    "Agent execution failed: %s",
                    error,
                )

                answer = (
                    "An error occurred while processing "
                    "your question."
                )

                sources = []

                st.error(
                    f"{answer}\n\nDetails: {error}"
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "chunks": result.get("chunks", []) if "result" in locals() else [],
        }
    )


# ---------------------------------------------------------------------------
# Existing ChromaDB inspection
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🔍 ChromaDB Inspection")

inspection_col1, inspection_col2 = st.columns(
    [1, 3]
)

with inspection_col1:
    inspection_limit = st.number_input(
        "Show first N chunks",
        min_value=5,
        max_value=100,
        value=15,
        step=5,
    )

    if st.button("Refresh inspection"):
        st.rerun()

with inspection_col2:
    inspection_stats = get_collection_stats()

    st.info(
        f"**Collection:** "
        f"`{inspection_stats.get('collection_name')}` · "
        f"**Count:** "
        f"{inspection_stats.get('document_count', 0)} · "
        f"**Embedding dimension:** "
        f"{inspection_stats.get('embedding_dimension', '—')}"
    )

inspection_items = inspect_collection(
    limit=int(inspection_limit)
)

if not inspection_items:
    st.warning(
        "No documents are currently stored in the collection. "
        "Upload and ingest study materials first."
    )

else:
    for item in inspection_items:
        item_id = item.get(
            "id",
            "Unknown ID",
        )

        with st.expander(
            f"ID: `{item_id}`"
        ):
            metadata = item.get(
                "metadata",
                {},
            )

            st.json(metadata)

            st.markdown("**Text preview**")

            st.text(
                item.get(
                    "text_preview",
                    "",
                )
            )

            if item.get("embedding_dim"):
                st.caption(
                    f"Embedding dimension: "
                    f"{item['embedding_dim']}"
                )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()

st.caption(
    "AI Study Assistant · Built with Streamlit, "
    "LangChain, ChromaDB & OpenRouter · "
    "Answers are strictly grounded in your "
    "uploaded materials."
)
