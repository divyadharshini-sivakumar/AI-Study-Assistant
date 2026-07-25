# AI Study Assistant

A complete, production-ready **multi-document RAG** study assistant built with **Streamlit**, **LangChain**, **ChromaDB**, local free embeddings, and a controlled **ReAct-style agent**.  
All LLM calls go through **OpenRouter** using a free model. No API keys are hard-coded.

---

## Features

- Upload multiple **PDF**, **TXT** and **CSV** study materials
- Rich metadata preserved for every chunk (source filename, file type, page, row, chunk ID)
- Local free embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- Persistent ChromaDB vector store
- Strict grounded answering – returns exactly  
  `The uploaded study materials do not contain enough information.` when evidence is insufficient
- Controlled ReAct agent (Decide → Retrieve → Validate → Answer)
- Study tools: Summary, Flashcards, MCQs, Interview Questions
- ChromaDB inspection UI (IDs, metadata, text previews, embedding dimension)
- One-click PDF project report generator
- Optimised for free-tier limits (caching, modest context, low temperature)

---

## Project Structure

```
AI-Study-Assistant/
├── app.py                 # Streamlit frontend
├── ingest.py              # Document loading, chunking, ingestion
├── rag.py                 # ChromaDB + retrieval + inspection
├── prompts.py             # All prompts (zero-shot, few-shot, role, etc.)
├── agent.py               # Controlled ReAct-style agent
├── utils.py               # Shared helpers, env, embeddings
├── report_generator.py    # PDF report generation
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── data/                  # Uploaded files land here
├── chroma_db/             # Persistent vector store
├── assets/
└── reports/               # Generated PDF reports
```

---

## Quick Start

### 1. Clone & environment

```bash
git clone <your-repo-url>
cd AI-Study-Assistant
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env
```

Edit `.env` and set:

```env
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct:free
```

Get a free key at [https://openrouter.ai/keys](https://openrouter.ai/keys).

### 3. Run

```bash
streamlit run app.py
```

Expected output: browser opens at `http://localhost:8501` with the Study Assistant UI.

---

## Usage Guide

1. **Upload** one or more PDF / TXT / CSV files in the sidebar.
2. Click **Ingest**. Chunks are embedded and stored in ChromaDB.
3. Ask questions in the chat. The agent decides whether retrieval is needed, validates relevance, and answers only from your materials.
4. Use the **Study Tools** dropdown to generate summaries, flashcards, MCQs or interview questions.
5. Open **ChromaDB Inspection** to explore stored chunks, metadata and embedding dimensions.
6. Click **Generate PDF Report** to produce a full project report (architecture, setup, sample outputs, deployment instructions, etc.).

---

## Prompt Engineering Techniques Used

| Technique              | Where it appears                          |
|------------------------|-------------------------------------------|
| Zero-shot              | Main QA, summary, interview questions     |
| Few-shot               | Flashcards & MCQs (explicit examples)     |
| Role prompting         | System role for the study tutor           |
| Contextual             | Retrieved chunks injected into every prompt |
| CoT-inspired           | Internal reasoning stages (never exposed) |
| Prompt chaining        | Agent stages: Decide → Validate → Answer  |

---

## Free-tier Optimisations

- Local embeddings → zero cost, offline after first download
- Small `CHUNK_SIZE` (800) and `TOP_K` (4)
- Low temperature (0.2) and modest `max_tokens`
- Session-state caching to avoid repeated OpenRouter calls
- Single-word decisions in the agent routing/validation stages
- Exact fallback sentence enforced at multiple points

If the free model is rate-limited, simply change `OPENROUTER_MODEL` in `.env` to another free model listed on OpenRouter.

---

## Streamlit Community Cloud Deployment

1. Push the repo to GitHub.
2. Go to [https://share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select repository, branch `main`, main file `app.py`.
4. Under **Secrets** add:

```toml
OPENROUTER_API_KEY = "sk-or-v1-..."
OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct:free"
```

5. Deploy. The first run downloads the embedding model (one-time).

---

## Testing Checklist

- [ ] Upload PDF + TXT + CSV → successful ingestion with correct metadata
- [ ] Ask a question answered by the materials → grounded answer + sources
- [ ] Ask something outside the materials → exact insufficient-information sentence
- [ ] Generate flashcards / MCQs → few-shot style respected
- [ ] Clear DB → collection emptied
- [ ] Rebuild from `data/` → chunks restored
- [ ] Chroma inspection → IDs, metadata, previews, embedding dimension visible
- [ ] PDF report generated and downloadable

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `OPENROUTER_API_KEY` missing | Copy `.env.example` → `.env` and fill the key |
| Embedding model download slow | First run only; subsequent runs are offline |
| Free model rate-limited | Switch to another free model on OpenRouter |
| Empty retrieval results | Lower `SIMILARITY_THRESHOLD` or increase `TOP_K` in `.env` |
| Chroma permission errors | Ensure `chroma_db/` is writable |

---

## License

MIT – feel free to use, modify and share for learning and teaching purposes.
