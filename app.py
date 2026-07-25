"""
app.py
------
AI Study Assistant – Streamlit frontend.
Modern, student-friendly UI with full RAG + ReAct agent capabilities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import streamlit as st

from utils import (
    init_session_state,
    clear_session_keys,
    DATA_DIR,
    get_collection_name,
)
from rag import (
    get_collection_stats,
    inspect_collection,
    clear_vectorstore,
)
from ingest import ingest_uploaded_files, rebuild_knowledge_base
from agent import (
    run_agent,
    generate_summary,
    generate_flashcards,
    generate_mcqs,
    generate_interview_questions,
)
from report_generator import generate_report

# ---------------------------------------------------------------------------
# Page config & logging
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
# Session state defaults
# ---------------------------------------------------------------------------

DEFAULTS = {
    "messages": [],
    "last_agent_result": None,
    "ingestion_log": None,
    "study_tool_output": None,
    "study_tool_type": None,
}

init_session_state(DEFAULTS)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📚 AI Study Assistant")
    st.caption("Multi-document RAG · ReAct Agent · Free-tier ready")

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
        ingest_btn = st.button("➕ Ingest", use_container_width=True, type="primary")
    with col2:
        clear_btn = st.button("🗑️ Clear DB", use_container_width=True)

    rebuild_btn = st.button("🔄 Rebuild from data/", use_container_width=True)

    if ingest_btn and uploaded_files:
        with st.spinner("Ingesting documents…"):
            result = ingest_uploaded_files(uploaded_files, clear_first=False)
            st.session_state.ingestion_log = result
            st.success(f"Added {result['total_chunks_added']} chunks from {result['files_processed']} file(s).")
            st.rerun()

    if clear_btn:
        clear_vectorstore()
        clear_session_keys(["messages", "last_agent_result", "study_tool_output"])
        st.session_state.ingestion_log = {"status": "cleared"}
        st.success("Vector database cleared.")
        st.rerun()

    if rebuild_btn:
        with st.spinner("Rebuilding knowledge base from data/ …"):
            result = rebuild_knowledge_base()
            st.session_state.ingestion_log = result
            st.success(f"Rebuilt: {result['total_chunks_added']} chunks.")
            st.rerun()

    # Live stats
    stats = get_collection_stats()
    st.divider()
    st.subheader("📊 Collection Stats")
    st.metric("Documents / Chunks", stats.get("document_count", 0))
    st.caption(f"Collection: `{stats.get('collection_name', '—')}`")
    if stats.get("embedding_dimension"):
        st.caption(f"Embedding dim: {stats['embedding_dimension']}")

    st.divider()
    st.subheader("🛠️ Study Tools")
    tool = st.selectbox(
        "Choose a tool",
        ["None", "Summary", "Flashcards", "MCQs", "Interview Questions"],
        index=0,
    )
    n_items = st.slider("Number of items", 3, 10, 5)

    if st.button("Generate", use_container_width=True) and tool != "None":
        if stats.get("document_count", 0) == 0:
            st.warning("Knowledge base is empty. Upload materials first.")
        else:
            # Use a broad query to pull representative chunks
            from rag import retrieve_relevant_only
            chunks = retrieve_relevant_only("key concepts definitions important points", top_k=8)
            if not chunks:
                st.warning("No relevant chunks found.")
            else:
                with st.spinner(f"Generating {tool.lower()}…"):
                    if tool == "Summary":
                        out = generate_summary(chunks)
                    elif tool == "Flashcards":
                        out = generate_flashcards(chunks, n=n_items)
                    elif tool == "MCQs":
                        out = generate_mcqs(chunks, n=n_items)
                    else:
                        out = generate_interview_questions(chunks, n=n_items)
                    st.session_state.study_tool_output = out
                    st.session_state.study_tool_type = tool

    st.divider()
    if st.button("📄 Generate PDF Report", use_container_width=True):
        with st.spinner("Building report…"):
            path = generate_report()
            st.success(f"Report saved to `{path.name}`")
            with open(path, "rb") as f:
                st.download_button(
                    "Download Report",
                    f,
                    file_name=path.name,
                    mime="application/pdf",
                    use_container_width=True,
                )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Ask your study materials")
st.caption("Answers are grounded exclusively in the documents you uploaded.")

# Ingestion feedback
if st.session_state.ingestion_log:
    with st.expander("Last ingestion details", expanded=False):
        st.json(st.session_state.ingestion_log)

# Study tool output
if st.session_state.study_tool_output:
    st.subheader(f"📝 {st.session_state.study_tool_type}")
    st.markdown(st.session_state.study_tool_output)
    st.divider()

# Chat interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

# User input
if prompt := st.chat_input("Ask a question about your materials…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = run_agent(prompt)
            st.session_state.last_agent_result = result
            answer = result["answer"]
            st.markdown(answer)

            sources = result.get("sources", [])
            if sources:
                with st.expander("Retrieved sources & relevance"):
                    for chunk in result.get("chunks", []):
                        score = chunk.get("score", 0)
                        st.markdown(
                            f"- {chunk['citation']}  \n"
                            f"  Relevance score (distance): `{score:.4f}` · "
                            f"{'✅ relevant' if chunk.get('is_relevant') else '⚠️ low relevance'}"
                        )
                        st.caption(chunk["page_content"][:250] + "…")

            # Show agent decision trail (lightweight transparency)
            with st.expander("Agent trace (ReAct stages)"):
                st.write(f"**Decision:** {result.get('decision')}")
                st.write(f"**Validation:** {result.get('validation')}")
                st.write(f"**Chunks retrieved:** {len(result.get('chunks', []))}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )

# ---------------------------------------------------------------------------
# ChromaDB Inspection
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🔍 ChromaDB Inspection")

col_a, col_b = st.columns([1, 3])
with col_a:
    limit = st.number_input("Show first N chunks", min_value=5, max_value=100, value=15, step=5)
    if st.button("Refresh inspection"):
        st.rerun()

with col_b:
    stats = get_collection_stats()
    st.info(
        f"**Collection:** `{stats.get('collection_name')}` · "
        f"**Count:** {stats.get('document_count', 0)} · "
        f"**Embedding dim:** {stats.get('embedding_dimension', '—')}"
    )

items = inspect_collection(limit=limit)
if not items:
    st.warning("No documents in the collection yet. Upload and ingest materials first.")
else:
    for item in items:
        with st.expander(f"ID: `{item['id']}`"):
            st.json(item["metadata"])
            st.markdown("**Text preview**")
            st.text(item["text_preview"])
            if item.get("embedding_dim"):
                st.caption(f"Embedding dimension: {item['embedding_dim']}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "AI Study Assistant · Built with Streamlit, LangChain, ChromaDB & OpenRouter · "
    "Answers are strictly grounded in your uploaded materials."
)
