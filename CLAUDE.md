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
PDF or markdown → pdfplumber (text + table markdown), or raw text
                → strip running headers, drop the table of contents
                → Document → Section → Clause  (parse.py)
                → one chunk per clause, never crossing a page break
                → local embeddings (intfloat/multilingual-e5-small, 512-token
                  window; the chunk budget is derived from that window)
                → ChromaDB

Query time:
Question → route by document family / role (e.g. S-737 TRS)
         → BM25 ∥ vector search → reciprocal rank fusion
         → local cross-encoder rerank (mmarco-mMiniLMv2, cached per process)
         → top 12 chunks → LLM returns a claims schema
         → every claim validated against the excerpts it cites
         → Streamlit UI
```

### Key decisions

**Everything retrieves locally and free.** Embeddings, BM25 and the reranker all
run on CPU. The only paid-ish call is the final answer, and on the Groq free
tier that is enough for a demo.

**Chunk size is derived, not chosen.** The chunk budget is computed from the
embedding model's own token window at ingest time. The original bug was 3,500-
character chunks against a 256-token model, which left 53% of the corpus
invisible to search.

**Multilingual by embedding, not by translation.** The corpus is embedded once
with a multilingual model, so a Bahasa question searches natively. This deleted
`langdetect`, `deep-translator`, a second vector store and the language-override
control — and fixed a Bahasa page hit-rate that was 0% before.

**Citations are verified, not requested.** The model returns a claims schema and
`answer.validate_claims()` rejects any claim citing an excerpt that was not
retrieved, or quoting text that is not in it. The prompt no longer has to be
trusted for correctness of citations.

**Providers are config, not code.** `llm.py` builds chat models from
`LLM_PROVIDER`, validates model ids at startup (Groq retires models, which is
how this app died) and falls back to the next model on a rate limit or a
retirement.

**Query expansion was removed.** It scored identically with and without it —
one LLM call per question for no measurable gain.

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

### Measured results

`python demo1_doc_intelligence/eval/run_eval.py --no-expand` — 18 questions
(12 English, 6 Bahasa), retrieval only:

| Stage | Right document | Right page @12 |
|---|---|---|
| Original (English model, 3,500-char chunks) | 94% | 56% |
| Clause chunks, still English model | 94% | 50% |
| + multilingual embeddings + mMARCO rerank | 100% | 78% |
| + hybrid BM25/RRF + document routing | **100%** | **89%** |

English 92%, Bahasa Indonesia 83% (was 0%). The full progression and the
reasoning behind each step are in
`demo1_doc_intelligence/docs/plans/2026-09-18-demo1-overhaul.md`.

### Pending

- Deploy to Streamlit Cloud (index builds on first run, so no index is committed)
- Add the live demo link to the README

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
├── demo1_doc_intelligence/
│   ├── README.md                          ← demo1 detail + troubleshooting
│   ├── requirements.txt                   ← demo1 only (demo2/3 differ)
│   ├── requirements-dev.txt               ← pytest
│   ├── .env                               ← gitignored, copy .env.example
│   ├── .env.example
│   ├── data/
│   │   ├── raw_docs/                      ← gitignored
│   │   ├── vectorstore_en/                ← gitignored
│   │   └── vectorstore_multi/             ← gitignored
│   ├── docs/plans/                        ← implementation plans
│   ├── tests/                             ← pytest smoke tests
│   └── src/
│       ├── ingest.py                      ← PDF → chunks → vectorstore
│       ├── retrieval.py                   ← question → rerank → answer
│       ├── config.py                      ← branding + LLM settings
│       └── app.py                         ← Streamlit UI
├── demo2_epci_data/
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
