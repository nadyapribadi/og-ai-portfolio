# CLAUDE.md — og-ai-portfolio

Three AI demos for upstream oil & gas. All use public data, zero-cost stack.

**Owner:** Nadya Boyke Pribadi
**Repo:** https://github.com/nadyapribadi/og-ai-portfolio

---

## Demos

| Demo | Status | Stack |
|------|--------|-------|
| `demo1_doc_intelligence` | 🔨 In progress | LangChain + ChromaDB + Groq + Streamlit |
| `demo2_incident_intelligence` | 📋 In development | LangChain SQL + SQLite + Streamlit |
| `demo3_ddr_assistant` | 📋 Planned | N8N + Groq + Streamlit |

---

## Personas (applies to all demos)

**Persona 1 — Site/HSE Engineer**
On site, time pressure, mobile device. Needs fast answers with citations they can verify.
Zero tolerance for friction, extra clicks, or ambiguous answers.

**Persona 2 — Procurement/Contracts Engineer**
Office-based. Daily pain: verifying JIP33 spec compliance.
Knows the vocabulary (deluge valve, LOPC, CAS levels).
Trusts the tool if it speaks their language and cites correctly.

**Persona 3 — IT/Digital Champion**
Evaluates whether to show this to their CIO.
Needs the tool to look like a product, not a student project.
Must be able to visibly demonstrate multilingual capability.

---

## Demo 1 — Current State

### Architecture

```
PDFs → pdfplumber (text + table markdown)
     → RecursiveCharacterTextSplitter (3500 chars, overlap 150)
     → filter chunks < 100 chars
     → HuggingFace embeddings (all-MiniLM-L6-v2) locally
     → ChromaDB vectorstore_en/

Query time:
Question → langdetect → Google Translate if non-English
         → expand_query() via Groq 8B → 3 search variants
         → MMR search across variants (fetch_k=30)
         → cross-encoder reranker (ms-marco-MiniLM-L-6-v2, local, ~80MB)
         → top 6 chunks → Groq 70B → answer with strict citation rules
         → Streamlit UI
```

### Key decisions

**No Reverse HyDE at ingest.** Removed — exhausted 100k daily Groq token quota for 275 chunks.
Cross-encoder reranker solves vocabulary mismatch without any API calls.

**pdfplumber over PyPDF.** Preserves table column structure. Critical for 456.pdf consequence tables.

**Dual model:**
- `llama-3.1-8b-instant` — query expansion (~150 tokens, fast)
- `llama-3.3-70b-versatile` — answering (~2000-3000 tokens, quality)

**Groq free tier:** 100k tokens/day on 70B. ~35-50 answers/day. Fine for demos.

**7-rule system prompt:**
1. Use only provided excerpts — no gap-filling from general knowledge
2. Cite every claim: (Source: filename, Page: N)
3. Prohibitions are complete answers
4. Never hedge falsely if answer is in excerpts
5. Keep technical terms in English even in Bahasa answers
6. Show incomplete lists honestly
7. Say "not found" only if genuinely absent

### Document pack

| File | Content |
|------|---------|
| 459.pdf | IOGP 459 — Life-Saving Rules |
| 456.pdf | IOGP 456 — Process Safety KPIs |
| S-737v2026-03 TRS.pdf | JIP33 S-737 Deluge Skids — Technical |
| S-737Qv2026-03 QRS.pdf | JIP33 S-737 Deluge Skids — Quality |
| S-717v2025-03 TRS.pdf | JIP33 S-717 Noise Equipment — Technical |
| S-717Qv2020-06 QRS.pdf | JIP33 S-717 Noise Equipment — Quality |
| S-719v2025-01 TRS.pdf | JIP33 S-719 Water Mist — Technical |
| S-719Qv2025-01 QRS.pdf | JIP33 S-719 Water Mist — Quality |
| S-719Jv2025-01 TRS with Justification.pdf | JIP33 S-719 — Justification |

### Verified working questions

English: life saving rules, confined space, hot work, Tier 1/2 KPIs,
LOPC consequences, S-737 electrical standards, deluge skid design,
suspended load prohibition, process safety KPI measurement.

Bahasa Indonesia: aturan keselamatan jiwa, ruang tertutup,
Tier 1 vs Tier 2, persyaratan desain deluge skid.

### Pending

- Test all questions through UI after daily token reset
- Deploy to Streamlit Cloud
- Add live demo link to README

---

## Demo 2 — Incident Intelligence Copilot

### Data sources

**Phase 1 (MVP):**
- BOEM offshore incident records (CSV/XLSX, public, row-level data)
- OSHA incident data (CSV, public, row-level data)

**Phase 2 (enrichment):**
- CSB investigation reports as RAG layer (PDFs only, not SQL-ready)
- Combined architecture: SQL (BOEM+OSHA) + RAG (CSB narratives)

**IOGP data:** Aggregated stats only (TRIR, fatalities by category). Use for benchmarking only.

### Database schema (Phase 1)

```sql
incidents (incident_id, date, operator, location, incident_type, severity, fatalities, injuries, description)
equipment (equipment_id, category, subcategory)
incident_equipment (incident_id, equipment_id)
root_causes (cause_id, category, description)
incident_causes (incident_id, cause_id)
corrective_actions (action_id, incident_id, description, category)
```

### AI features
1. Text-to-SQL — plain English to structured queries
2. Trend analysis — time series, year-over-year
3. Root cause intelligence — top causes by incident type
4. Recommendation engine — corrective actions from historical patterns

### Stack
SQLite (MVP) → LangChain SQLChain → Groq 70B → Streamlit + Plotly

---

## Demo 3 — DDR Drafting Assistant

Engineer fills structured form → N8N triggers Groq 70B → DDR narrative drafted →
Engineer reviews and approves in Streamlit → exported.

Human-in-the-loop by design. Not full automation — augmentation.

Stack: N8N (SumoPod hosting) + Groq LLaMA 3.3 70B + Streamlit

---

## Environment

```
Python: 3.11.9 (pyenv)
OS: macOS (Apple Silicon MBP)
venv: ~/og-ai-portfolio/venv/
```

### API keys (.env, never commit)
```
GROQ_API_KEY=gsk_...
```

### Groq free tier
- llama-3.3-70b-versatile: 100k tokens/day, 6k tokens/minute
- llama-3.1-8b-instant: 500k tokens/day, 20k tokens/minute

---

## File structure

```
og-ai-portfolio/
├── CLAUDE.md
├── README.md
├── .env                                   ← gitignored
├── requirements.txt
├── demo1_doc_intelligence/
│   ├── data/
│   │   ├── raw_docs/                      ← gitignored
│   │   ├── vectorstore_en/                ← gitignored
│   │   └── vectorstore_multi/             ← gitignored
│   └── src/
│       ├── ingest.py                      ← PDF → chunks → vectorstore
│       ├── retrieval.py                   ← question → rerank → answer
│       └── app.py                         ← Streamlit UI
├── demo2_incident_intelligence/
│   └── src/
│       ├── download_data.py
│       ├── build_database.py
│       ├── text_to_sql.py
│       └── app.py
└── demo3_ddr_assistant/
    └── src/
        ├── ddr_generator.py
        └── app.py
```

---

## Domain vocabulary

| Term | Meaning |
|------|---------|
| IOGP | International Association of Oil & Gas Producers |
| JIP33 | Joint Industry Programme 33 — standardized procurement specs |
| LOPC | Loss of Primary Containment |
| PSE | Process Safety Event |
| TRS / QRS / IRS / PDS | Technical / Quality / Information Requirements Spec / Product Data Sheet |
| EPCI | Engineering, Procurement, Construction, Installation |
| DDR | Daily Drilling Report |
| IADC | International Association of Drilling Contractors |
| BOEM | Bureau of Ocean Energy Management |
| TRIR | Total Recordable Incident Rate |
| CAS | Conformity Assessment System |
| SIL | Safety Integrity Level |
| NOC / IOC | National / International Oil Company |

---

## Constraints (never violate)

- No confidential data — public documents and federal datasets only
- No INPEX branding or references
- Answers must cite source document and page number
- "Not found" is always better than a wrong answer
- Embedding and reranking must remain local — no API cost