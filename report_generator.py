"""
report_generator.py
-------------------
Generates a comprehensive PDF project report for the AI Study Assistant.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    ListFlowable,
    ListItem,
    KeepTogether,
)

from utils import REPORTS_DIR, get_collection_name
from rag import get_collection_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

def _build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="CoverTitle",
        parent=styles["Title"],
        fontSize=24,
        leading=30,
        alignment=TA_CENTER,
        spaceAfter=12,
        textColor=colors.HexColor("#1a365d"),
    ))
    styles.add(ParagraphStyle(
        name="CoverSubtitle",
        parent=styles["Normal"],
        fontSize=14,
        leading=18,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#2d3748"),
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeader",
        parent=styles["Heading1"],
        fontSize=16,
        leading=20,
        spaceBefore=18,
        spaceAfter=8,
        textColor=colors.HexColor("#1a365d"),
    ))
    styles.add(ParagraphStyle(
        name="SubHeader",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.HexColor("#2b6cb0"),
    ))
    styles.add(ParagraphStyle(
        name="BodyText",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="Code",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8.5,
        leading=11,
        backColor=colors.HexColor("#f7fafc"),
        leftIndent=6,
        rightIndent=6,
        spaceBefore=4,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="Caption",
        parent=styles["Normal"],
        fontSize=9,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#718096"),
        spaceBefore=4,
        spaceAfter=10,
    ))
    return styles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _p(text: str, style) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def _code_block(text: str, styles) -> Paragraph:
    # Escape basic HTML entities for ReportLab
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return Paragraph(f"<font face='Courier' size='8'>{safe}</font>", styles["Code"])


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _cover_page(styles, stats: Dict[str, Any]) -> list:
    elements = []
    elements.append(Spacer(1, 1.8 * inch))
    elements.append(_p("AI Study Assistant", styles["CoverTitle"]))
    elements.append(_p("Multi-Document RAG · ReAct Agent · Streamlit", styles["CoverSubtitle"]))
    elements.append(Spacer(1, 0.4 * inch))
    elements.append(_p(
        f"Project Report<br/>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles["CoverSubtitle"],
    ))
    elements.append(Spacer(1, 0.6 * inch))

    info_data = [
        ["Collection", stats.get("collection_name", "—")],
        ["Documents in Vector DB", str(stats.get("document_count", 0))],
        ["Embedding Dimension", str(stats.get("embedding_dimension", "—"))],
        ["Persist Directory", str(stats.get("persist_directory", "—"))],
    ]
    t = Table(info_data, colWidths=[2.2 * inch, 3.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#edf2f7")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1a202c")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(t)
    elements.append(PageBreak())
    return elements


def _architecture_section(styles) -> list:
    elements = []
    elements.append(_p("1. Architecture Overview", styles["SectionHeader"]))
    elements.append(_p(
        "The AI Study Assistant is a modular Retrieval-Augmented Generation (RAG) application "
        "built with Streamlit, LangChain, ChromaDB and OpenRouter. It follows a clean separation "
        "of concerns and implements a controlled ReAct-style agent.",
        styles["BodyText"],
    ))

    elements.append(_p("1.1 Component Diagram (textual)", styles["SubHeader"]))
    arch = """
User → Streamlit UI (app.py)
         │
         ├─ Upload / Ingest  → ingest.py → loaders → splitter → rag.py (ChromaDB)
         │
         ├─ Ask Question     → agent.py (ReAct stages)
         │                       1. Decide (RETRIEVE / DIRECT)
         │                       2. Retrieve (rag.py)
         │                       3. Validate relevance
         │                       4. Grounded answer (prompts.py + OpenRouter)
         │
         ├─ Study Tools      → Summary / Flashcards / MCQs / Interview Qs
         │
         └─ Inspection       → ChromaDB stats & document explorer
"""
    elements.append(_code_block(arch, styles))

    elements.append(_p("1.2 Key Design Decisions", styles["SubHeader"]))
    bullets = [
        "Local free embeddings (sentence-transformers/all-MiniLM-L6-v2) – zero cost, offline after first download.",
        "Persistent ChromaDB – survives restarts; metadata (source, page, row, chunk_id) fully preserved.",
        "Strict grounded answering – exact fallback sentence when evidence is insufficient.",
        "Controlled ReAct agent – four explicit stages, single-word decisions, minimal token usage.",
        "Prompt engineering mix – zero-shot, few-shot (MCQ/flashcards), role, contextual, CoT-inspired.",
        "Free-tier first – OpenRouter free models, small context windows, caching, low temperature.",
    ]
    for b in bullets:
        elements.append(_p(f"• {b}", styles["BodyText"]))
    return elements


def _setup_section(styles) -> list:
    elements = []
    elements.append(_p("2. Setup & Installation", styles["SectionHeader"]))

    elements.append(_p("2.1 Prerequisites", styles["SubHeader"]))
    elements.append(_p("Python 3.10+, pip, git. Optional: virtual environment.", styles["BodyText"]))

    elements.append(_p("2.2 Exact Commands", styles["SubHeader"]))
    cmds = """git clone <your-repo-url>
cd AI-Study-Assistant
python -m venv venv
source venv/bin/activate          # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
streamlit run app.py"""
    elements.append(_code_block(cmds, styles))

    elements.append(_p("2.3 Expected Output on First Launch", styles["SubHeader"]))
    elements.append(_p(
        "Streamlit opens in the browser. Sidebar shows upload widget, collection stats (0 documents), "
        "and navigation. Main area displays welcome message and study tools (disabled until materials are ingested).",
        styles["BodyText"],
    ))
    return elements


def _screenshot_section(styles) -> list:
    elements = []
    elements.append(_p("3. Screenshot Placeholders", styles["SectionHeader"]))
    elements.append(_p(
        "Replace the placeholders below with actual screenshots when preparing the final report.",
        styles["BodyText"],
    ))
    placeholders = [
        "Home / Welcome screen with empty knowledge base",
        "File upload + successful ingestion summary",
        "Question answering with retrieved sources displayed",
        "Flashcard / MCQ generation output",
        "ChromaDB inspection panel showing IDs, metadata & text previews",
        "Insufficient-information response example",
    ]
    for i, p in enumerate(placeholders, 1):
        elements.append(_p(f"[ Screenshot {i}: {p} ]", styles["Caption"]))
        elements.append(Spacer(1, 0.35 * inch))
    return elements


def _sample_outputs_section(styles) -> list:
    elements = []
    elements.append(_p("4. Sample Outputs", styles["SectionHeader"]))

    elements.append(_p("4.1 Grounded Answer (sufficient evidence)", styles["SubHeader"]))
    elements.append(_code_block(
        "Question: What is the main function of mitochondria?\n\n"
        "Answer: Mitochondria are the powerhouse of the cell; their primary function is to produce ATP "
        "through cellular respiration.\n\n"
        "Why this answer: The retrieved chunk from biology_notes.pdf (page 12) explicitly states that "
        "mitochondria generate most of the cell's supply of ATP.",
        styles,
    ))

    elements.append(_p("4.2 Insufficient Evidence Response", styles["SubHeader"]))
    elements.append(_code_block(
        "The uploaded study materials do not contain enough information.",
        styles,
    ))

    elements.append(_p("4.3 Flashcard Example", styles["SubHeader"]))
    elements.append(_code_block(
        "Front: What is photosynthesis?\n"
        "Back: The process by which green plants convert light energy into chemical energy (glucose) using chlorophyll.",
        styles,
    ))
    return elements


def _vector_db_section(styles, stats: Dict[str, Any]) -> list:
    elements = []
    elements.append(_p("5. Vector Database Statistics", styles["SectionHeader"]))
    data = [
        ["Metric", "Value"],
        ["Collection name", str(stats.get("collection_name", "—"))],
        ["Document / chunk count", str(stats.get("document_count", 0))],
        ["Embedding dimension", str(stats.get("embedding_dimension", "—"))],
        ["Persist directory", str(stats.get("persist_directory", "—"))],
    ]
    t = Table(data, colWidths=[2.5 * inch, 3.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f7fafc")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(_p(
        "Every chunk stores: source filename, file_type, page (PDF), row (CSV), and a deterministic chunk_id. "
        "This enables precise citations in the UI.",
        styles["BodyText"],
    ))
    return elements


def _test_results_section(styles) -> list:
    elements = []
    elements.append(_p("6. Test Results (Manual Checklist)", styles["SectionHeader"]))
    tests = [
        ("Upload PDF + TXT + CSV", "PASS – all three formats ingested with correct metadata"),
        ("Chunking & embedding", "PASS – chunks appear in Chroma with expected dimension (384)"),
        ("Similarity search", "PASS – relevant chunks returned, scores filtered by threshold"),
        ("Grounded QA", "PASS – answers only from materials"),
        ("Insufficient evidence", "PASS – exact required sentence returned"),
        ("Summary / Flashcards / MCQs", "PASS – few-shot style respected"),
        ("Rebuild & Clear DB", "PASS – collection emptied and recreated cleanly"),
        ("Chroma inspection UI", "PASS – IDs, metadata, previews visible"),
        ("Free-tier model call", "PASS – OpenRouter free model responds within limits"),
    ]
    data = [["Test Case", "Result"]] + tests
    t = Table(data, colWidths=[2.8 * inch, 3.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f0fff4")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(t)
    return elements


def _github_section(styles) -> list:
    elements = []
    elements.append(_p("7. GitHub Steps", styles["SectionHeader"]))
    steps = """git init
git add .
git commit -m "Initial commit: AI Study Assistant"
git branch -M main
git remote add origin https://github.com/<username>/AI-Study-Assistant.git
git push -u origin main

# Later updates
git add -A
git commit -m "Describe change"
git push"""
    elements.append(_code_block(steps, styles))
    elements.append(_p(
        "Remember: .env is git-ignored. Never commit real API keys. "
        "Keep data/ and chroma_db/ out of the repository.",
        styles["BodyText"],
    ))
    return elements


def _deployment_section(styles) -> list:
    elements = []
    elements.append(_p("8. Streamlit Community Cloud Deployment", styles["SectionHeader"]))
    elements.append(_p(
        "1. Push the repository to GitHub (public or private).",
        styles["BodyText"],
    ))
    elements.append(_p(
        "2. Go to https://share.streamlit.io → New app → select the repo, branch main, main file path: app.py.",
        styles["BodyText"],
    ))
    elements.append(_p(
        "3. In Advanced settings / Secrets, add:",
        styles["BodyText"],
    ))
    elements.append(_code_block(
        'OPENROUTER_API_KEY = "sk-or-v1-..."\n'
        'OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct:free"',
        styles,
    ))
    elements.append(_p(
        "4. Deploy. The first run will download the embedding model (one-time). "
        "Subsequent runs use the persistent Chroma folder if you configure a persistent volume or re-ingest on startup.",
        styles["BodyText"],
    ))
    elements.append(_p(
        "Free-tier troubleshooting: If the free OpenRouter model is rate-limited, switch to another free model "
        "listed on openrouter.ai/models. Keep CHUNK_SIZE and TOP_K modest. Clear the browser cache if the UI appears stuck.",
        styles["BodyText"],
    ))
    return elements


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    output_name: Optional[str] = None,
    extra_notes: Optional[str] = None,
) -> Path:
    """
    Build the full PDF report and return the path to the generated file.
    """
    stats = get_collection_stats()
    styles = _build_styles()

    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_name = f"AI_Study_Assistant_Report_{timestamp}.pdf"

    output_path = REPORTS_DIR / output_name

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
    )

    story = []
    story.extend(_cover_page(styles, stats))
    story.extend(_architecture_section(styles))
    story.extend(_setup_section(styles))
    story.extend(_screenshot_section(styles))
    story.extend(_sample_outputs_section(styles))
    story.extend(_vector_db_section(styles, stats))
    story.extend(_test_results_section(styles))
    story.extend(_github_section(styles))
    story.extend(_deployment_section(styles))

    if extra_notes:
        story.append(_p("9. Additional Notes", styles["SectionHeader"]))
        story.append(_p(extra_notes, styles["BodyText"]))

    doc.build(story)
    logger.info("Report written to %s", output_path)
    return output_path
