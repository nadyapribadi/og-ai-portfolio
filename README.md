# Upstream O&G Document Intelligence

> AI solutions across the upstream oil & gas value chain — from procurement intelligence to EPCI project controls and field operations automation.

Built by **Nadya Boyke Pribadi** — AI Solution Architect specializing in upstream O&G digitalization.

[![GitHub](https://img.shields.io/badge/GitHub-nadyapribadi-blue)](https://github.com/nadyapribadi)

---

## Three Demos

| # | Demo | Problem solved | Stack | Status |
|---|------|---------------|-------|--------|
| 1 | [Doc Intelligence](#demo-1--document-intelligence) | Query upstream O&G standards in plain English | LangChain · ChromaDB · Groq · Streamlit | 🔨 Building |
| 2 | [EPCI Data Assistant](#demo-2--epci-data-assistant) | Ask your GR/GI project data in plain English | LangChain SQL · SQLite · Streamlit | 📋 Planned |
| 3 | [DDR Drafting Assistant](#demo-3--ddr-drafting-assistant) | Auto-draft Daily Drilling Reports from structured inputs | N8N · Groq · Streamlit | 📋 Planned |

---

## Demo 1 — Document Intelligence

**Problem:** Procurement engineers and HSE leads spend hours searching through IOGP standards, JIP33 specifications, and process safety guidelines to answer routine questions.

**Solution:** A RAG chatbot grounded in 9 public O&G documents. Ask in English or Bahasa Indonesia — get cited answers with source document and page number.

**What makes it different from a generic chatbot:**
- Domain-specific system prompt with 7 strict rules — no hallucination, no gap-filling
- Reverse HyDE at ingest — 5 questions generated per chunk for better retrieval
- Dual model — Groq 70B for quality answers, 8B for fast query expansion
- Multilingual — auto-detects Bahasa Indonesia, translates for search, answers in original language
- pdfplumber extraction — preserves table structure (critical for consequence tables)

### Document sources (all public, all free)

| Document | Source |
|----------|--------|
| IOGP Report 459 — Life-Saving Rules | iogp.org |
| IOGP Report 456 — Process Safety KPIs | iogp.org |
| IOGP JIP33 S-737 — Deluge Skids (TRS + QRS) | iogp.org |
| IOGP JIP33 S-717 — Noise Emitting Equipment (TRS + QRS) | iogp.org |
| IOGP JIP33 S-719 — Water Mist Fire Protection (TRS + QRS + Justification) | iogp.org |

### Architecture

```
PDFs (pdfplumber) → chunks (3500 chars) → Reverse HyDE questions (Groq 70B)
→ ChromaDB (local, sentence-transformers embeddings)
→ Query: langdetect → translate → expand_query (Groq 8B) → MMR search
→ Answer: Groq 70B (llama-3.3-70b-versatile) with strict citation rules
→ Streamlit UI with streaming, clickable questions, source citations
```

---

## Demo 2 — EPCI Data Assistant

**Problem:** Project controls engineers on EPCI projects struggle to get quick answers from GR/GI data — "What's outstanding for Contractor X?", "Which bulk materials are overdue?"

**Solution:** Natural language interface over a synthetic EPCI project database. Text-to-SQL over realistic GR/GI, WBS, contractor, and materials data.

**Synthetic data model:**
- Materials table (bulk, tagged, piping)
- Goods Receipt (GR) table with WBS codes, contractor, quantity, date
- Goods Issue (GI) table linked to construction activities
- Contractors table
- WBS table (3.1 Civil, 3.2 Mechanical, 3.3 Electrical, 3.4 Instrumentation)

---

## Demo 3 — DDR Drafting Assistant

**Problem:** Drilling engineers spend 30-45 min/day writing Daily Drilling Reports manually — repetitive narrative from data that already exists.

**Solution:** Engineer fills a structured form → N8N pipeline → Groq LLM generates IADC-standard DDR narrative → human reviews and approves.

Human-in-the-loop by design — not a replacement, a drafting assistant.

---

## Setup

```bash
git clone https://github.com/nadyapribadi/og-ai-portfolio
cd og-ai-portfolio

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Add your GROQ_API_KEY (free at console.groq.com)
```

### Demo 1 — Run locally

```bash
# Step 1: Add PDFs to demo1_doc_intelligence/data/raw_docs/
# (download IOGP 459, 456, S-717, S-719, S-737 from iogp.org)

# Step 2: Ingest (builds vectorstore, ~10 min first time)
python demo1_doc_intelligence/src/ingest.py

# Step 3: Run chatbot
streamlit run demo1_doc_intelligence/src/app.py
```

---

## Zero-cost stack

| Component | Tool | Cost |
|-----------|------|------|
| LLM (answering) | Groq llama-3.3-70b-versatile | Free |
| LLM (query expansion) | Groq llama-3.1-8b-instant | Free |
| Embeddings | sentence-transformers (local) | Free |
| Vector DB | ChromaDB (local) | Free |
| PDF extraction | pdfplumber | Free |
| UI | Streamlit Community Cloud | Free |
| N8N (Demo 3) | SumoPod | ~$1/month |

---

## Architecture diagrams

See [`/docs/architecture/`](./docs/architecture/) — Mermaid diagrams for each demo pipeline.

---

*Built as a pre-bootcamp portfolio. November 2026 — Purwadhika AI Engineering bootcamp.*
