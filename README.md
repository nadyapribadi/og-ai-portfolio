# Upstream O&G AI Portfolio

Three working AI demos for upstream oil & gas operations.
Built by **Nadya Boyke Pribadi**.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/nadya-nadya-404309104)
[![GitHub](https://img.shields.io/badge/GitHub-nadyapribadi-black)](https://github.com/nadyapribadi)

---

## Demos

| | Demo | Problem Solved | Status |
|---|---|---|---|
| 1 | [O&G Document Intelligence](#demo-1--og-document-intelligence) | Search IOGP standards and JIP33 specs in plain English or Bahasa Indonesia | ✅ Working |
| 2 | [Incident Intelligence Copilot](#demo-2--incident-intelligence-copilot) | Query offshore and industrial incident patterns | 📋 In development |
| 3 | [Drilling Operations Copilot](#demo-3--drilling-operations-copilot) | Auto-draft Daily Drilling Reports from structured inputs | 📋 Planned |

---

## Demo 1 — O&G Document Intelligence

Ask questions from IOGP standards and JIP33 specifications in plain English or Bahasa Indonesia.
Get cited answers with source document and page number.

**Try it:** *(link when deployed)*

**Documents indexed:**
- IOGP 459 — Life-Saving Rules
- IOGP 456 — Process Safety KPIs
- JIP33 S-737 — Deluge Skids (TRS + QRS)
- JIP33 S-717 — Noise Equipment (TRS + QRS)
- JIP33 S-719 — Water Mist Fire Protection (TRS + QRS + Justification)

**In scope:**
- Text-based PDFs (digitally created)
- Technical documents — standards, specifications, procedures, guidelines
- English and Bahasa Indonesia
- Up to ~50 documents, ~500 pages total

**Out of scope:**
- Scanned PDFs — answers will be "not found"
- Image-heavy documents — engineering drawings, P&IDs, charts
- Questions requiring real-time data or calculations

→ See [demo1_doc_intelligence/demo1_README.md](demo1_doc_intelligence/demo1_README.md) for troubleshooting, customization, and tips.

---

## Demo 2 — Incident Intelligence Copilot

*(In development — details to follow)*

---

## Demo 3 — Drilling Operations Copilot

*(Planned — details to follow)*

---

## Run Demo 1 Locally

```bash
git clone https://github.com/nadyapribadi/og-ai-portfolio
cd og-ai-portfolio
python -m venv venv && source venv/bin/activate
pip install -r demo1_doc_intelligence/requirements.txt

# Add your Groq key — free at console.groq.com
cp demo1_doc_intelligence/.env.example demo1_doc_intelligence/.env
# then paste your key into that file

# Add your PDFs (see demo1_README.md for the document pack)
mkdir -p demo1_doc_intelligence/data/raw_docs

python demo1_doc_intelligence/src/ingest.py   # ~30 seconds
streamlit run demo1_doc_intelligence/src/app.py
```

You can skip adding PDFs: if `data/raw_docs/` is empty the app indexes the small
**sample corpus** in `demo1_doc_intelligence/sample_docs/` instead, so a fresh
clone works immediately. The app builds its index on first run if one is
missing, which is what makes it deployable anywhere without committing an index.

Run the tests:

```bash
pip install -r demo1_doc_intelligence/requirements-dev.txt
python -m pytest demo1_doc_intelligence/tests -q
```

---

## Deploy Demo 1

Nothing has to be committed to deploy: the app builds its index on first run
from whatever documents are present, starting with the bundled sample corpus.

**Streamlit Community Cloud** — free, and the natural host for a Streamlit app:

1. [share.streamlit.io](https://share.streamlit.io) → *New app* → pick this repo
   and branch.
2. **Main file path:** `demo1_doc_intelligence/src/app.py`
3. **Advanced settings → Secrets:**
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
4. Deploy.

The root `requirements.txt` exists only as a deployment shim — Streamlit Cloud
looks for it at the repository root, while each demo keeps its own dependencies
inside its own folder.

The first load takes a minute or two: it downloads the embedding model (~470 MB)
and builds the index, both cached for the life of the container.

**Not Netlify.** Streamlit needs a long-running Python process and a WebSocket
per user; Netlify serves static files and short-lived functions. Netlify is the
right home for a portfolio landing page that links to the demo.

---

## Stack

LangChain · ChromaDB · Groq LLaMA 3.3 70B · sentence-transformers · pdfplumber · Streamlit · N8N
