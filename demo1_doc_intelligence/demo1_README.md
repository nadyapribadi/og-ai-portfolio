# Demo 1 — O&G Document Intelligence

RAG chatbot over public upstream O&G documents.
English and Bahasa Indonesia supported.

---

## Status

| File | Status |
|------|--------|
| `ingest.py` | ✅ Working — 30 seconds, no API calls |
| `retrieval.py` | ✅ Working — cross-encoder reranker |
| `app.py` | ✅ Working — Streamlit UI with streaming |

---

## In Scope

- Text-based PDFs (digitally created, not scanned)
- Technical documents — standards, specifications, procedures, guidelines
- English and Bahasa Indonesia
- Up to ~50 documents, ~500 pages total recommended

## Out of Scope

- Scanned PDFs — pdfplumber extracts nothing, all answers will be "not found"
- Image-heavy pages — engineering drawings, P&IDs, charts, figures
- Questions requiring cross-document comparison ("compare X in doc A with Y in doc B")
- Real-time data or calculations
- Questions where the answer is only in a table spanning multiple pages

---

## Document Pack (default)

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

Replace or add any text-based PDFs in `data/raw_docs/` and re-run ingest.

---

## Verified Questions (all passing)

**English:**
- "What are the life saving rules?"
- "What are the 9 IOGP life saving rules?"
- "What must I confirm before entering a confined space?"
- "What are the hot work requirements in a hazardous area?"
- "What is the difference between Tier 1 and Tier 2 process safety events?"
- "How are process safety KPIs measured?"
- "What does LOPC stand for and what are its consequences?"
- "What standards does S-737 reference for electrical installations?"
- "What must workers do before walking under a suspended load?"
- "What does IOGP S-737 specify for deluge skid design?"

**Bahasa Indonesia:**
- "Apa saja aturan keselamatan jiwa?"
- "Apa yang harus dilakukan sebelum memasuki ruang tertutup?"
- "Apa perbedaan antara kejadian keselamatan proses Tier 1 dan Tier 2?"
- "Apa persyaratan desain deluge skid menurut IOGP S-737?"

---

## Architecture

```
PDFs → pdfplumber (text + table markdown)
     → RecursiveCharacterTextSplitter (3500 chars, overlap 150)
     → filter chunks < 100 chars
     → HuggingFace embeddings (all-MiniLM-L6-v2, local)
     → ChromaDB vectorstore

Query:
Question → langdetect → Google Translate if non-English
         → expand_query (Groq 8B, 3 variants)
         → MMR search (fetch_k=30)
         → cross-encoder reranker (ms-marco-MiniLM-L-6-v2, local)
         → top 6 chunks → Groq 70B → answer with citations
```

**Key decisions:**
- No question generation at ingest — cross-encoder reranker handles vocabulary mismatch
- pdfplumber extracts tables as markdown — preserves column structure for consequence tables
- Groq 70B for answering, 8B for query expansion — quality where it matters, speed where it doesn't

---

## Tips

- Specific questions get better answers than vague ones
  - ✅ "What are the confined space entry requirements?"
  - ⚠️ "Tell me about confined spaces"
- If you get "not found", try rephrasing with different vocabulary
- The system works best with documents that have clear structure — headings, numbered sections
- Language is auto-detected — just type in English or Bahasa Indonesia naturally
- Use the language override in the sidebar if auto-detection is wrong

---

## Customizing Sample Questions

If you replace the default documents with your own, update the sample questions in `app.py`.

Find the `SAMPLE_QUESTIONS` dict (around line 110):

```python
SAMPLE_QUESTIONS = {
    "🦺 HSE Rules": [
        "What are the life saving rules?",
        ...
    ],
    ...
}
```

Replace with categories and questions relevant to your documents.

---

## Verify Your Index

After running ingest, confirm your documents were indexed correctly:

```bash
python3 -c "
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
vs = Chroma(
    collection_name='og_docs',
    embedding_function=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'),
    persist_directory='demo1_doc_intelligence/data/vectorstore_en'
)
print(f'Chunks indexed: {vs._collection.count()}')
results = vs.similarity_search('what is this document about', k=3)
for r in results:
    print(r.metadata.get('source_file'), 'page', r.metadata.get('page'))
    print(r.page_content[:150])
    print()
"
```

**What to check:**
- Chunk count should be roughly 20-30 chunks per document
- Content preview should show readable text — not garbled characters
- If chunk count is 0 or very low → likely scanned PDFs (see troubleshooting)

---

## Troubleshooting

**"Not found" for everything**
Your PDFs are likely scanned. Verify with:
```bash
pdftotext yourfile.pdf - | head -20
```
If output is empty or garbled, the PDF has no text layer.
Solution: use OCR software to convert scanned PDFs to text-based PDFs before ingest.

**Chunk count is very low after ingest**
Same issue — pdfplumber extracted nothing from scanned pages.

**Rate limit error in the app**
Groq free tier: 100,000 tokens/day on 70B model (~35-50 answers/day).
Wait a few minutes — limits reset in rolling windows, not just at midnight.

**Answer is correct but missing details**
The answer is limited to what was retrieved. Try rephrasing the question.
If consistently incomplete, the relevant content may be in a complex table or figure
that pdfplumber could not extract cleanly.

**Wrong language in answer**
Use the language dropdown in the sidebar to override auto-detection.

---

## Known Limitations

| Limitation | Root cause |
|---|---|
| Scanned PDFs return nothing | pdfplumber requires a text layer |
| Complex multi-page tables may be incomplete | pdfplumber reads page by page |
| 35-50 questions/day on free Groq tier | 100k token/day limit on 70B model |
| Author/metadata questions often fail | Author info is on page 1, semantically far from metadata queries |

---

## File Locations

```
demo1_doc_intelligence/
├── data/
│   ├── raw_docs/          ← PDFs go here (gitignored)
│   ├── vectorstore_en/    ← ChromaDB (gitignored, rebuilt by ingest.py)
│   └── vectorstore_multi/ ← ChromaDB multilingual (gitignored, disabled)
└── src/
    ├── ingest.py          ← PDF → chunks → vectorstore (~30 seconds)
    ├── retrieval.py       ← question → rerank → answer
    └── app.py             ← Streamlit UI
```