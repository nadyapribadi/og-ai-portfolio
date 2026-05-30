# Upstream O&G AI Portfolio

Three working AI demos for upstream oil & gas operations.
Built by **Nadya Boyke Pribadi**.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/nadya-nadya-404309104)
[![GitHub](https://img.shields.io/badge/GitHub-nadyapribadi-black)](https://github.com/nadyapribadi)

---

## Demos

| | Demo | Problem Solved | Status |
|---|---|---|---|
| 1 | [O&G Document Intelligence](#demo-1--og-document-intelligence) | Search IOGP standards and JIP33 specs in plain English | 🔨 Building |
| 2 | [Incident Intelligence Copilot](#demo-2--incident-intelligence-copilot) | Query offshore and industrial incident patterns | 📋 In development |
| 3 | [Drilling Operations Copilot](#demo-3--drilling-operations-copilot) | Auto-draft Daily Drilling Reports from structured inputs | 📋 Planned |

---

## Demo 1 — O&G Document Intelligence

Ask questions from 9 public IOGP standards and JIP33 specifications.
Get cited answers with source document and page number. English and Bahasa Indonesia supported.

**Try it:** *(link when deployed)*

**Sample questions:**
- "What are the 9 IOGP Life-Saving Rules?"
- "What must I confirm before entering a confined space?"
- "What does S-737 specify for deluge valve inspection?"
- "What is the difference between Tier 1 and Tier 2 process safety events?"
- "Apa saja aturan keselamatan jiwa menurut IOGP?"

**Documents:**
- IOGP 459 — Life-Saving Rules
- IOGP 456 — Process Safety KPIs
- JIP33 S-737 — Deluge Skids (TRS + QRS)
- JIP33 S-717 — Noise Equipment (TRS + QRS)
- JIP33 S-719 — Water Mist Fire Protection (TRS + QRS + Justification)

---

## Demo 2 — Incident Intelligence Copilot

Ask questions about real offshore and industrial incident records in plain English.
Get trend analysis, root cause breakdowns, and corrective action patterns.

**Data sources:**
- BOEM — Bureau of Ocean Energy Management (offshore incidents)
- OSHA — Occupational Safety and Health Administration

**Sample questions:**
- "What are the most common causes of confined space fatalities offshore?"
- "Show lifting incident trends by year."
- "Which equipment categories have the highest fatality rates?"

---

## Demo 3 — Drilling Operations Copilot

Engineer fills a structured form → AI drafts the IADC DDR narrative → Engineer reviews and approves.

---

## Run Demo 1 Locally

```bash
git clone https://github.com/nadyapribadi/og-ai-portfolio
cd og-ai-portfolio
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Add GROQ_API_KEY to .env (free at console.groq.com)
# Add PDFs to demo1_doc_intelligence/data/raw_docs/

python demo1_doc_intelligence/src/ingest.py   # ~30 seconds
streamlit run demo1_doc_intelligence/src/app.py
```

---

## Stack

LangChain · ChromaDB · Groq LLaMA 3.3 70B · sentence-transformers · pdfplumber · Streamlit · N8N